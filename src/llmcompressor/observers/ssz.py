from typing import Optional

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize, quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.utils import patch_attr

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["SSZObserver"]

@Observer.register("ssz")
class SSZObserver(Observer):
    """
    SSZ (Scan-Scale-Zero) weight quantization observer.

    Optimizes scale and offset parameters iteratively using least squares
    to minimize quantization MSE. Uses MinMax initialization followed by
    SSZ iterative refinement.

    Supported: CHANNEL strategy, int8/int4, symmetric and asymmetric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.iter_num = observer_kwargs.get("iter_num", 50)
        self.threshold = observer_kwargs.get("threshold", 1e-10)
        self.min_scale = observer_kwargs.get("min_scale", 1e-30)
    
    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        global_scale = self._get_module_param("global_scale")
        return _ssz_calculate(
            observed.T,
            self.args,
            self.iter_num,
            self.threshold,
            self.min_scale,
            global_scale=global_scale,
        )
    

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _ssz_calculate(
            observed.T,
            self.args,
            self.iter_num,
            self.threshold,
            self.min_scale,
            global_scale=None,
        )
    

def _ssz_calculate(
    observed: torch.Tensor,
    args: QuantizationArgs,
    iter_num: int,
    threshold: float,
    min_scale: float,
    global_scale: Optional[torch.Tensor] = None,
) -> MinMaxTuple:
    """
    Compute quantization parameters using SSZ (Scan-Scale-Zero) algorithm.

    Args:
        observed: Tensor to compute quantization parameters for.
        args: QuantizationArgs containing quantization configuration.
        iter_num: Maximum number of iterations for optimization.
        threshold: Convergence threshold for optimization.
        min_scale: Minimum allowed scale value to prevent underflow.
        global_scale: Optional precomputed global scale to use during optimization.

    Returns:
        Tuple of (min_vals, max_vals) representing the optimized quantization range.
    """
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    best_scale, best_zero_point = calculate_qparams(
        min_vals, max_vals, args, global_scale
    )

    best_quant_weight = quantize(
        observed,
        best_scale,
        best_zero_point,
        args,
        global_scale=global_scale if args.strategy == "tensor_group" else None,
    )

    best_dequant_weight = fake_quantize(
        observed,
        best_scale,
        best_zero_point,
        args,
        global_scale=global_scale if args.strategy == "tensor_group" else None,
    )

    best_mse = torch.mean(torch.pow(torch.abs((observed - best_dequant_weight)), 2), dim=0, keepdim=True)

    quant_weight = best_quant_weight

    for i in range(iter_num):
        if args.symmetric:
            current_scale = (torch.sum(observed * quant_weight, dim=0, keepdim=True) /
                             torch.sum(quant_weight * quant_weight, dim=0, keepdim=True)
                             .clamp(min=min_scale))
            current_zero_point = best_zero_point
            quant_weight = quantize(
                observed,
                current_scale,
                current_zero_point,
                args,
                global_scale=global_scale if args.strategy == "tensor_group" else None,
            )
            
        else:
            quant_weight_zero_point = quant_weight - best_zero_point
            current_scale = (torch.sum(observed * quant_weight_zero_point, dim=0, keepdim=True) /
                             torch.sum(quant_weight_zero_point * quant_weight_zero_point, dim=0, keepdim=True)
                             .clamp(min=min_scale))
            
            current_zero_point = (torch.sum(quant_weight_zero_point * current_scale - observed, dim=0, keepdim=True) /
                                    (observed.shape[0] * current_scale))
            quant_weight = quantize(
                observed,
                current_scale,
                current_zero_point,
                args,
                global_scale=global_scale if args.strategy == "tensor_group" else None,
            )
        
        current_dequant_weight = fake_quantize(
            observed,
            current_scale,
            current_zero_point,
            args,
            global_scale=global_scale if args.strategy == "tensor_group" else None,
        )

        current_mse = torch.mean(torch.pow(torch.abs((observed - current_dequant_weight)), 2),
                                 dim=0, keepdim=True).squeeze()

        mask1 = (best_mse - current_mse) / best_mse.clamp(min=1e-4) < threshold
        mask2 = torch.abs(best_mse - current_mse) < threshold

        if (torch.sum(torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))) == 0):
            break

        mask = (current_mse < best_mse).to(torch.int32)
        best_mse = best_mse * (1 - mask) + current_mse * mask
        best_scale = (best_scale * (1 - mask) + current_scale * mask).squeeze()

        if args.symmetric:
            best_zero_point = current_zero_point
        else:
            best_zero_point = (best_zero_point * (1 - mask) + current_zero_point * mask).squeeze()

        best_quant_weight = best_quant_weight * (1 - mask) + quant_weight * mask
    return best_scale, best_zero_point

