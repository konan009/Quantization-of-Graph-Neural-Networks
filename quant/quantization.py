import torch
from torch.autograd.function import InplaceFunction
import torch.nn as nn
from collections import namedtuple


        
def get_qparams(max_val, min_val, num_bits):
    max_val, min_val = float(max_val), float(min_val)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)

    qmin = -(2.0 ** (num_bits - 1)) 
    qmax = qmin + 2.0 ** num_bits - 1

    if max_val == min_val:
        scale = 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / float(qmax - qmin)
        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
        zero_point = zero_point

    return qmin, qmax, zero_point, scale

    
class Quantize(InplaceFunction):
    @classmethod
    def forward(
        cls, ctx, input, num_bits, ste, min_val, max_val
    ):
        
        output = input.clone()

        # compute qparams
        qmin, qmax, zero_point, scale = get_qparams(
            max_val, min_val, num_bits
        )

        ctx.STE = ste # save stuff for backprop (if STE not enabled)
        if not ste:
            ctx.save_for_backward(input)
            ctx.qmin = qmin
            ctx.qmax = qmax
            ctx.scale = scale
            ctx.zp = zero_point

        inv_scale = 1.0 / scale

        output.mul_(inv_scale).add_(zero_point)
        output.round_().clamp_(qmin, qmax)  # quantize
        output.add_(-zero_point).mul_(scale)  # dequantize

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.STE == "Vanilla":
            return grad_output, None, None, None, None, None, None, None
        elif ctx.STE == "Clipping":
            return grad_output, None, None, None, None, None, None, None # TO BE EDIT
        else:
            return grad_output, None, None, None, None, None, None, None # TO BE EDIT

    
quantize = Quantize.apply


class Quantizer(nn.Module):
    def __init__(
        self,
        quant_type: str,
        calibration: str,
        ste: str,
    ):
        super(Quantizer, self).__init__()
        self.quant_type = quant_type
        self.ste = ste
        self.calibration = calibration
        
        if(self.quant_type=="INT8"):
            self.num_bits = 8
        elif(self.quant_type=="INT4"):
            self.num_bits = 4
        else:
            self.num_bits = 8 # DEFAULT


        if(calibration=="Min/Max"):
            self.min_fn = torch.min  
            self.max_fn = torch.max
        elif(calibration=="percent"):
            self.min_fn = torch.min    # TO BE EDIT PERCENTAGE MAPPING
            self.max_fn = torch.max
        elif(calibration=="entropy"):
            self.min_fn = torch.min    # TO BE EDIT TO KL DIVERGENCE
            self.max_fn = torch.max

    def update_ranges(self, input):
        min = self.min_fn(input)
        max = self.max_fn(input)

        self.min_val = min
        self.max_val = max


    def forward(self,input):
        self.update_ranges(input.detach())
        return quantize(
            input,
            self.num_bits,
            self.ste,
            self.min_val,
            self.max_val
        )


