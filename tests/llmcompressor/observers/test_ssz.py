import pytest
import torch
from compressed_tensors.quantization import fake_quantize
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import SSZObserver, Observer


@pytest.mark.parametrize(
    "strategy,symmetric,exp_loss",
    [
        ("tensor", True, 4.8103e-06),
        ("tensor", False, 1.1258e-06),
        ("channel", True, 2.5675e-06),
        ("channel", False, 2.3696e-07),
        ("group", True, 3.1282e-06),
        ("group", False, 1.3794e-07),
        ("block", True, 2.8968e-06),
        ("block", False, 5.6068e-07),
    ],
)
def test_mse_observer(strategy, symmetric, exp_loss):
    tensor = torch.arange(24).reshape((6, 4)) / 24
    num_bits = 8
    weights = QuantizationArgs(
        num_bits=num_bits,
        strategy=strategy,
        symmetric=symmetric,
        group_size=(2 if strategy == "group" else None),
        block_structure=([3, 2] if strategy == "block" else None),
        observer="ssz",
    )

    observer = weights.observer
    observer = Observer.load_from_registry(observer, base_name="weight", args=weights)
    assert isinstance(observer, SSZObserver)

    scale, zero_point = observer(tensor)
    q_tensor = fake_quantize(tensor, scale, zero_point, weights)
    mse_loss = torch.sum((tensor - q_tensor).abs_().pow_(2)) / tensor.numel()
    assert mse_loss == pytest.approx(exp_loss, abs=1e-10)
