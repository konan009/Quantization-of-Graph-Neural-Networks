
import torch
from torch.autograd.function import InplaceFunction
from torch.nn import Parameter, Module, ModuleDict
from collections import namedtuple
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add
from collections import OrderedDict
import inspect
from constants import REQUIRED_QUANTIZER_KEYS,REQUIRED_GCN_KEYS
from torch_geometric.utils import (
    softmax,
    add_self_loops,
    remove_self_loops,
    add_remaining_self_loops,
)



def scatter_(name, src, index, dim=0, dim_size=None):
    """Taken from an earlier version of PyG"""
    assert name in ["add", "mean", "min", "max"]

    op = getattr(torch_scatter, "scatter_{}".format(name))
    out = op(src, index, dim, None, dim_size)              
    out = out[0] if isinstance(out, tuple) else out

    if name == "max":
        out[out < -10000] = 0
    elif name == "min":
        out[out > 10000] = 0

    return out



class MessagePassingQuant(Module):
    """Modified from the PyTorch Geometric message passing class"""

    def __init__(
        self, aggr="add", flow="source_to_target", node_dim=0, mp_quantizers=None
    ):
        super(MessagePassingQuant, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__args__ = set().union(msg_args, aggr_args, update_args)

        assert mp_quantizers is not None
        self.mp_quant_fns = mp_quantizers

    def reset_parameters(self):
        self.mp_quantizers = ModuleDict()
        for key in REQUIRED_QUANTIZER_KEYS:
            self.mp_quantizers[key] = self.mp_quant_fns[key]()

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (
                    f"Encountered node tensor with size "
                    f"{tensor.size(self.node_dim)} in dimension {self.node_dim}, "
                    f"but expected size {size[index]}."
                )
            )

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out["edge_index"] = edge_index
        out["edge_index_i"] = edge_index[i]
        out["edge_index_j"] = edge_index[j]
        out["size"] = size
        out["size_i"] = size[i]
        out["size_j"] = size[j]

        # Add special aggregate arguments.
        out["index"] = out["edge_index_i"]
        out["dim_size"] = out["size_i"]

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"Required parameter {key} is empty.")
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2
        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.mp_quantizers["message"](self.message(**msg_kwargs))

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        out = self.mp_quantizers["aggregate"](self.aggregate(out, **aggr_kwargs))

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.mp_quantizers["update_q"](self.update(out, **update_kwargs))

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def update(self, inputs):  # pragma: no cover
        return inputs


class GCNConvQuant(MessagePassingQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        bias=True,
        normalize=True,
        mp_quantizers=None,
        layer_quantizers=None,
        **kwargs,
    ):
        super(GCNConvQuant, self).__init__(aggr="add", mp_quantizers=mp_quantizers, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in REQUIRED_GCN_KEYS:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):

        # quantizing input
        x_q = self.layer_quantizers["inputs"](x)

        # quantizing layer weights
        w_q = self.layer_quantizers["weights"](self.weight)
        x = torch.matmul(x_q, w_q)

        x = self.layer_quantizers["features"](x)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(
                    edge_index,
                    x.size(self.node_dim),
                    edge_weight,
                    self.improved,
                    x.dtype,
                )
            else:
                norm = edge_weight

            norm = self.layer_quantizers["norm"](norm)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )