from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generator, Iterator, Sequence
from weakref import WeakKeyDictionary

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tqdm import tqdm


@dataclass
class IntermediateValue:
    """
    Dataclass which recursively defines offloaded values and which device to onload to

    :param value: either an offloaded Tensor, an primative value, or a recursable value
    :param device: if the value is a Tensor, then the device to onload the tensor to,
        otherwise None
    """

    value: torch.Tensor | "IntermediateValue" | Any
    device: torch.device | None


class IntermediatesCache:
    """
    Cache which stores intermediate values (activations) produced by batched, sequential
    execution of models. Values are offloaded to the `offload_device` when stored in
    the cache and onloaded to their original device when fetched from the cache. If
    `offload_device` is None, values will not be offloaded at all.

    Currently supports nested offloading of dataclass instances and tuples

    Construct using `empty` and `from_dataloader` class methods
    """

    # map of onload value -> offload value
    # used to avoid excess memory usage when shared tensors are offloaded
    offload_values: WeakKeyDictionary[torch.Tensor, torch.Tensor] = WeakKeyDictionary()

    def __init__(
        self,
        intermediate: IntermediateValue | None = None,
        offload_device: torch.device | None = "cpu",
    ):
        self.intermediate = intermediate
        self.offload_device = offload_device

    @classmethod
    def empty(cls, offload_device: torch.device | None = "cpu"):
        """
        Construct an empty cache

        :param offload_device: device to offload values to
        """
        return cls(intermediate=None, offload_device=offload_device)

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        model_device: torch.device = torch.device("cpu"),
        offload_device: torch.device | None = torch.device("cpu"),
    ) -> list[IntermediatesCache]:
        """
        Initialize a list of cache with data from the provided dataloader

        This method iterates through all batches in the dataloader and offloads
        them to the specified device. For faster cache preparation, consider:
        - Increasing batch_size to reduce the number of iterations
        - Using num_workers > 0 in the DataLoader for parallel loading (e.g. the
          calibration DataLoader from format_calibration_data uses
          dataloader_num_workers; when > 0, pin_memory and prefetch_factor are
          also set where applicable, which speeds both cache build and calibration)
        - Ensuring data preprocessing is done before creating the dataloader

        :param dataloader: dataloader which generates values to be cached
        :param model_device: device which values will be onloaded to when fetched
        :param offload_device: device to offload values to
        """
        batch_intermediates = [
            cls(cls._offload_value(batch, offload_device, model_device), offload_device)
            for batch in tqdm(dataloader, desc="Preparing cache")
        ]

        return batch_intermediates

    def fetch(self) -> Any:
        """
        Fetch the original value represented by the intermediate, onloading any tensors
        """
        if self.intermediate is None:
            raise ValueError("No intermediate to fetch")
        return self._onload_value(self.intermediate)

    def update(self, intermediate: Any):
        """
        Update/put the intermediate, offloading any tensors in the value.
        """
        self.intermediate = self._offload_value(intermediate, self.offload_device)

    def delete(self):
        """
        Delete the intermediate from the cache.
        """
        self.intermediate = None

    @classmethod
    def _onload_value(cls, intermediate: IntermediateValue) -> Any:
        """
        Onload a value's tensors to the onload device

        :param intermediate: intermediates value representation to onload
        :return: original value with tensors onloaded to the onload device
        """
        value = intermediate.value
        device = intermediate.device

        match value:
            case torch.Tensor():
                # use non_blocking when source is pinned and target is CUDA so the
                # H2D DMA can overlap with GPU compute on a separate CUDA stream
                non_blocking = (
                    value.is_pinned()
                    and device is not None
                    and torch.device(device).type == "cuda"
                )
                return value.to(device=device, non_blocking=non_blocking)
            case list():
                return [cls._onload_value(v) for v in value]
            case tuple():
                return tuple(cls._onload_value(v) for v in value)
            case dict():
                return {k: cls._onload_value(v) for k, v in value.items()}
            case _ if is_dataclass(value):
                for field in fields(value):
                    v = getattr(value, field.name)
                    setattr(value, field.name, cls._onload_value(v))
                return value
            case _:
                # handles primitive values that should be returned as is.
                # without this, a MatchError would be raised for unhandled types.
                return value

    @classmethod
    def _offload_value(
        cls,
        value: Any,
        offload_device: torch.device | None,
        onload_device: torch.device | None = None,
    ) -> IntermediateValue:
        """
        Offload a value's tensors to the offload device

        :param value: value to offload
        :param offload_device: device to offload `torch.Tensor` values to
        :param onload_device: device used when onloading `torch.Tensor` values.
            If None is provided, use the tensor's current device
        :return: Instance of IntermediateValue representing the offloaded value
        """
        kwargs = {"offload_device": offload_device, "onload_device": onload_device}
        match value:
            case torch.Tensor():
                with OverrideEqMode():
                    # check for cache hit between shared tensors
                    if value in cls.offload_values:
                        offloaded = cls.offload_values[value]
                    else:
                        # move to offload if no hit
                        offloaded = value.to(device=offload_device)
                        if offloaded is not value:  # avoid circular ref
                            # pin CPU tensors so onload can use non_blocking DMA
                            if (
                                torch.device(offload_device).type == "cpu"
                                and torch.cuda.is_available()
                                and not offloaded.is_pinned()
                            ):
                                offloaded = offloaded.pin_memory()
                            cls.offload_values[value] = offloaded

                return IntermediateValue(
                    value=offloaded,
                    device=(onload_device if onload_device else value.device),
                )
            case list():
                return IntermediateValue(
                    value=[cls._offload_value(v, **kwargs) for v in value],
                    device=None,
                )
            case tuple():
                return IntermediateValue(
                    value=tuple(cls._offload_value(v, **kwargs) for v in value),
                    device=None,
                )
            case dict():
                return IntermediateValue(
                    value={
                        k: cls._offload_value(v, **kwargs) for k, v in value.items()
                    },
                    device=None,
                )
            case _ if is_dataclass(value):
                for field in fields(value):
                    v = getattr(value, field.name)
                    setattr(value, field.name, cls._offload_value(v, **kwargs))
                return IntermediateValue(value=value, device=None)
            case _:
                # handles primitive values and provides a warning for unsupported types.
                # without this, values trigger a MatchError exception.
                if not isinstance(
                    value,
                    (int, str, float, bool, torch.dtype, torch.device, type(None)),
                ):
                    warnings.warn(f"Offloading not implemented for type {type(value)}.")
                return IntermediateValue(value=value, device=None)


def maybe_prefetch(batches: Sequence[IntermediatesCache]) -> Iterator[Any]:
    """
    Iterate with optional one-item background prefetch, controlled by
    ``active_session().state.sequential_prefetch``.
    Iterate over batches with the next batch prefetched in a background thread.
    Overlaps onload from offload_device with consumption of the current batch,
    which can reduce wall-clock time when offloading to CPU.
    When CUDA is available, uses non_blocking transfers (requires pinned CPU
    tensors, set up by _offload_value) and synchronises via CUDA events so the
    main stream waits for each H2D copy before running GPU kernels on the data.
    Yields the same fetched batch dicts as ; only the timing
    of onloads differs.
    """
    try:
        from llmcompressor.core import active_session

        use_prefetch = active_session().state.sequential_prefetch
    except Exception:
        use_prefetch = False

    if use_prefetch:
        # Single ThreadPoolExecutor for all caches
        yield from _prefetch_all(batches)
    else:
        # Direct fetch - replace each cache with its fetched value
        for batch in batches:
            yield batch.fetch()


def _prefetch_all(batches: Sequence[IntermediatesCache]) -> Generator[Any, None, None]:
    """Prefetch all caches in a single ThreadPoolExecutor."""

    # Create a dedicated CUDA stream for H2D transfers so they run on a
    # separate stream from the main thread's compute stream. Without this,
    # both threads default to the null stream (stream 0) which serializes
    # all operations and prevents any overlap.
    h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _fetch_and_record(batch):
        event = None
        if h2d_stream is not None:
            with torch.cuda.stream(h2d_stream):
                data = batch.fetch()
            event = torch.cuda.Event()
            event.record(h2d_stream)
        else:
            data = batch.fetch()
        return data, event

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for batch_index, batch in enumerate(batches):
            if future is not None:
                current, event = future.result()
            else:
                current, event = _fetch_and_record(batch)
            if batch_index + 1 < len(batches):
                future = executor.submit(_fetch_and_record, batches[batch_index + 1])
            else:
                future = None
            # Make the main CUDA stream wait for the background H2D copy
            # before any GPU kernel consumes the prefetched tensors
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            yield current


class OverrideEqMode(TorchDispatchMode):
    """
    When using a torch.Tensor as a key in a dictionary, the equality
    check must return a single value instead of a torch.Tensor
    of bool values.
    Use this override context for such cases, to swap out the torch.eq
    equality check for a check on id
    >>> a = torch.tensor([1,2,3])
    >>> b = torch.tensor([1,2,3])
    >>> a == b
    tensor([True, True, True])
    >>> with OverrideEqMode():
    ...     a == b
    tensor(True)
    """

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Check if the operation is equality
        if func is torch.ops.aten.eq.Tensor:
            # Override to use torch.equal
            assert len(args) == 2, "Exactly 2 args must be provided"

            # NOTE: Errors out without cast to torch.tensor
            return torch.tensor(id(args[0]) == id(args[1]))

        # For all other operations, just run them normally
        return func(*args, **kwargs)
