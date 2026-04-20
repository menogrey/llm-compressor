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
def test_ssz_observer(strategy, symmetric, exp_loss):
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

    torch.set_printoptions(precision=16)

    scale, zero_point = observer(tensor)
    print(scale, zero_point)
    q_tensor = fake_quantize(tensor, scale, zero_point, weights)
    mse_loss = torch.sum((tensor - q_tensor).abs_().pow_(2)) / tensor.numel()
    assert mse_loss == pytest.approx(exp_loss, abs=1e-10)


def _create_base_quantization_args(num_bits, strategy, symmetric, group_size):
    """Helper to create base QuantizationArgs without observer field."""
    return QuantizationArgs(
        num_bits=num_bits,
        strategy=strategy,
        symmetric=symmetric,
        group_size=group_size,
    )


def _run_observer_test(
    tensor, observer_name, strategy, symmetric, num_bits, group_size, module=None
):
    """
    Helper function to run observer and compute quantization error.

    Returns: (scale, zero_point, quantized_tensor, mse, global_scale)
    """
    weights = _create_base_quantization_args(num_bits, strategy, symmetric, group_size)
    weights.observer = observer_name

    observer = Observer.load_from_registry(
        observer_name, base_name="weight", args=weights, module=module
    )

    global_scale = None
    if strategy == "tensor_group" and module is not None:
        global_scale = observer.get_global_scale(tensor)
        module.weight_global_scale = global_scale

    scale, zero_point = observer(tensor)
    assert (scale >= 0).all(), "Scale values should be non-negative"

    weights_clean = _create_base_quantization_args(
        num_bits, strategy, symmetric, group_size
    )
    quantized = fake_quantize(
        tensor,
        scale,
        zero_point,
        weights_clean,
        global_scale=global_scale if strategy == "tensor_group" else None,
    )
    mse = torch.nn.functional.mse_loss(quantized, tensor)

    return scale, zero_point, quantized, mse, global_scale


def _assert_mse_comparison(
    mse_mse, ssz_mse, strategy, symmetric, is_real_weights=False
):
    """Assert MSE observer performance with appropriate slack."""
    epsilon = 1e-8
    slack = 1.20 if is_real_weights else 1.10

    if strategy == "tensor" and symmetric:
        assert mse_mse <= ssz_mse + epsilon, (
            f"MSE observer performed worse than SSZ observer!\n"
            f"Strategy: {strategy}, Symmetric: {symmetric}\n"
            f"SSZ MSE: {ssz_mse.item():.6e}\n"
            f"MSE Observer MSE: {mse_mse.item():.6e}\n"
            f"Difference: {(mse_mse - ssz_mse).item():.6e}"
        )
    else:
        assert mse_mse <= ssz_mse * slack + epsilon, (
            f"MSE observer performed significantly worse than SSZ observer!\n"
            f"Strategy: {strategy}, Symmetric: {symmetric}\n"
            f"SSZ MSE: {ssz_mse.item():.6e}\n"
            f"MSE Observer MSE: {mse_mse.item():.6e}\n"
            f"Difference: {(mse_mse - ssz_mse).item():.6e}\n"
            f"Ratio: {(mse_mse / (ssz_mse + epsilon)).item():.4f}x"
        )

@pytest.mark.parametrize(
    "strategy,symmetric,num_bits",
    [
        ("tensor", True, 8),
        ("tensor", False, 8),
        ("channel", True, 8),
        ("channel", False, 8),
        # ("tensor_group", True, 4),
        # ("tensor_group", False, 4),
        ("channel", True, 4),
        ("channel", False, 4),
    ],
)
@pytest.mark.parametrize(
    "std",
    [0.05, 0.2, 1.0],
    ids=["narrow", "medium", "wide"],
)
def test_mse_vs_ssz_on_random_tensor(strategy, symmetric, num_bits, std):
    """Test MSE observer error <= SSZ observer error on random tensors."""
    torch.manual_seed(42)
    tensor = torch.randn(128, 256) * std

    group_size = 32 if strategy == "tensor_group" else None

    module_ssz = None
    module_mse = None
    if strategy == "tensor_group":
        module_ssz = torch.nn.Linear(256, 128)
        module_ssz.weight.data = tensor
        module_mse = torch.nn.Linear(256, 128)
        module_mse.weight.data = tensor

    _, _, _, ssz_mse, _ = _run_observer_test(
        tensor,
        "ssz",
        strategy,
        symmetric,
        num_bits,
        group_size,
        module_ssz,
    )

    _, _, _, mse_mse, _ = _run_observer_test(
        tensor,
        "memoryless_mse",
        strategy,
        symmetric,
        num_bits,
        group_size,
        module_mse,
    )

    _assert_mse_comparison(
        mse_mse, ssz_mse, strategy, symmetric, is_real_weights=False
    )