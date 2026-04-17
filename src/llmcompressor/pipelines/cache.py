from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from weakref import WeakKeyDictionary

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tqdm import tqdm
from typing import TypeVar, Generic


T = TypeVar("T")


class IntermediatesCache(Generic[T]):
    """
    Cache which recursively defines offloaded values and which device to onload to.
    Recursive class can be used for fine-grained updates of data at each level.
    This cache stores intermediate values (activations) produced by batched, sequential
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
        intermediate: Any | None = None,
        onload_device: torch.device | None = None,
        offload_device: torch.device | None = "cpu",
    ):
        self.intermediate = intermediate
        self.offload_device = offload_device
        self.onload_device = onload_device

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
    ) -> IntermediatesCache:
        """
        Initialize a cache with data from the provided dataloader

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
            {
                key: cls._offload_value(value, offload_device, model_device)
                for key, value in batch.items()
            }
            for batch in tqdm(dataloader, desc="Preparing cache")
        ]

        cache = cls.empty(offload_device=offload_device)
        cache.update(batch_intermediates)
        return cache

    def fetch(self) -> T:
        """
        Fetch the original value represented by the intermediate, onloading any tensors
        """
        if self.intermediate is None:
            raise ValueError("No intermediate to fetch")
        return self._onload_value(self)

    def update(self, intermediate: T):
        """
        Update/put the intermediate, offloading any tensors in the value.
        """
        self.intermediate = self._offload_value(intermediate, self.offload_device).intermediate

    def clear(self):
        """
        Clear the intermediate from the cache.
        """
        self.intermediate = None

    def iter(self) -> Generator[Any, None, None]:
        """
        Iterate and onload batches from the cache. If the intermediate is a list/tuple,
        yields each item; otherwise yields the single intermediate value.
        """
        if self.intermediate is None:
            raise ValueError("No intermediate to fetch")
        
        value = self.intermediate
        if isinstance(value, (list, tuple)):
            for item in value:
                yield self._onload_value(item)
        else:
            yield self._onload_value(self)

    def iter_prefetch(self) -> Generator[Any, None, None]:
        """
        Iterate over batches with the next batch prefetched in a background thread.
        Overlaps onload from offload_device with consumption of the current batch,
        which can reduce wall-clock time when offloading to CPU.

        When CUDA is available, uses non_blocking transfers (requires pinned CPU
        tensors, set up by _offload_value) and synchronises via CUDA events so the
        main stream waits for each H2D copy before running GPU kernels on the data.

        Yields the same fetched batch dicts as :meth:`iter`; only the timing
        of onloads differs.
        """
        if self.intermediate is None:
            raise ValueError("No intermediate to fetch")
        
        value = self.intermediate
        if not isinstance(value, (list, tuple)):
            yield from self.iter()
            return

        num_batches = len(value)
        if num_batches == 0:
            return

        # Create a dedicated CUDA stream for H2D transfers so they run on a
        # separate stream from the main thread's compute stream. Without this,
        # both threads default to the null stream (stream 0) which serializes
        # all operations and prevents any overlap.
        h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def _fetch_and_record(batch_index):
            event = None
            if h2d_stream is not None:
                with torch.cuda.stream(h2d_stream):
                    data = self._onload_value(self.intermediate[batch_index])
                event = torch.cuda.Event()
                event.record(h2d_stream)
            else:
                data = self._onload_value(self.intermediate[batch_index])
            return data, event

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            for batch_index in range(num_batches):
                if future is not None:
                    current, event = future.result()
                else:
                    current, event = _fetch_and_record(batch_index)
                if batch_index + 1 < num_batches:
                    future = executor.submit(_fetch_and_record, batch_index + 1)
                else:
                    future = None
                # Make the main CUDA stream wait for the background H2D copy
                # before any GPU kernel consumes the prefetched tensors
                if event is not None:
                    torch.cuda.current_stream().wait_event(event)
                yield current

    def __iter__(self) -> Generator[Any, None, None]:
        yield from self.iter()

    def __len__(self) -> int:
        if self.intermediate is None:
            return 0

        if isinstance(self.intermediate, (list, tuple)):
            return len(self.intermediate)
        return 1
    
    def __getitem__(self, key) -> IntermediatesCache:
        """
        For recursive access to the intermediate value. If the intermediate is a
        container (eg: dict, list), allows access to the items within the container.
        Avoids the need to fetch the entire intermediate value when only a specific 
        item is needed. Note that this does not trigger an onload; the returned item
        is still in its offloaded state. Use the `onloaded` context manager for 
        onloading and editing specific items within the cache.
        """
        if self.intermediate is None:
            raise ValueError("No intermediate to get item from")
        return self.intermediate[key]
    
    def __setitem__(self, key, value):
        if self.intermediate is None:
            raise ValueError("No intermediate to set item on")
        self.intermediate[key] = self._offload_value(value, self.offload_device)

    def __delitem__(self, key):
        if self.intermediate is None:
            raise ValueError("No intermediate to delete item from")
        del self.intermediate[key]

    def append(self, value):
        if self.intermediate is None:
            raise ValueError("No intermediate to append to")
        if not isinstance(self.intermediate, list):
            raise TypeError("Intermediate is not a list, cannot append")
        self.intermediate.append(self._offload_value(value, self.offload_device))

    @contextmanager
    def onloaded(self, *path) -> Generator[Any, None, None]:
        """
        Context manager to edit a value at a specific path within the cache.

        On entry: navigates to and onloads only the target, yielding it.
        On exit: offloads the edited value back to the cache, without modifying
        other items in the same container.

        Example::
            with cache.onloaded(1, "foo") as item:
                item.copy_(new_tensor)  # offload on exit

            with cache.onloaded(1) as lst:
                lst.append(new_tensor)  # offload on exit (structure changed)

        :param path: index/key path to the target value
        :raises IndexError/KeyError: if path does not exist
        :raises ValueError: if no intermediate is cached
        """
        if self.intermediate is None:
            raise ValueError("No intermediate to edit")

        # Navigate to parent container
        parent = self.intermediate
        for key in path[:-1]:
            parent = parent[key]

        final_key = path[-1]
        target = parent[final_key]

        # Onload just the target
        offload_device = target.offload_device if isinstance(target, IntermediatesCache) else self.offload_device
        onloaded = self._onload_value(target)

        try:
            yield onloaded
        finally:
            parent[final_key] = self._offload_value(onloaded, offload_device)

    @classmethod
    def _onload_value(cls, cache: IntermediatesCache) -> Any:
        """
        Onload a value's tensors to the onload device

        :param intermediate: intermediates value representation to onload
        :return: original value with tensors onloaded to the onload device
        """
        value = cache.intermediate
        onload_device = cache.onload_device
        match value:
            case IntermediatesCache():
                return cls._onload_value(value)
            case torch.Tensor():
                # use non_blocking when source is pinned and target is CUDA so the
                # H2D DMA can overlap with GPU compute on a separate CUDA stream
                non_blocking = (
                    value.is_pinned()
                    and value.device is not None
                    and torch.device(value.device).type == "cuda"
                )
                return value.to(device=onload_device, non_blocking=non_blocking)
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
    ) -> IntermediatesCache:
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
            case IntermediatesCache():
                return cls._offload_value(value.intermediate, **kwargs)
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

                return IntermediatesCache(
                    offloaded,
                    onload_device=(onload_device if onload_device else value.device),
                    offload_device=offload_device,
                )
            case list():
                return IntermediatesCache(
                    [cls._offload_value(v, **kwargs) for v in value],
                    offload_device=offload_device,
                )
            case tuple():
                return IntermediatesCache(
                    tuple(cls._offload_value(v, **kwargs) for v in value),
                    offload_device=offload_device,
                )
            case dict():
                return IntermediatesCache(
                    {k: cls._offload_value(v, **kwargs) for k, v in value.items()},
                    offload_device=offload_device,
                )
            case _ if is_dataclass(value):
                for field in fields(value):
                    v = getattr(value, field.name)
                    setattr(value, field.name, cls._offload_value(v, **kwargs))
                return IntermediatesCache(value, offload_device=offload_device)
            case _:
                # handles primitive values and provides a warning for unsupported types.
                # without this, values trigger a MatchError exception.
                if not isinstance(
                    value,
                    (int, str, float, bool, torch.dtype, torch.device, type(None)),
                ):
                    warnings.warn(f"Offloading not implemented for type {type(value)}.")
                return IntermediatesCache(value, offload_device=offload_device)


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
