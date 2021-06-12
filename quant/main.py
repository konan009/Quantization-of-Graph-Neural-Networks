import torch

from quantization import Quantizer

def create_quantizer(quant_type = "INT8", ste = "Vanilla", calibration="Min/Max"):
    return Quantizer(
        quant_type=quant_type,
        calibration=calibration,
        ste=ste,
    )


quant_type = "INT8"
ste = "Vanilla"
calibration = "Min/Max"

layer_quantizers = {
    "inputs": create_quantizer(
        quant_type = quant_type, ste = ste, calibration=calibration
    ),
    "weights": create_quantizer(
        quant_type = quant_type, ste = ste, calibration=calibration
    ),
    "features": create_quantizer(
        quant_type = quant_type, ste = ste, calibration=calibration
    ),
}


torch.manual_seed(1)
random_float = torch.rand([10,10])
print(random_float)
print(layer_quantizers["inputs"](random_float))