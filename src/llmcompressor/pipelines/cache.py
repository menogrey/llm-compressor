from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from typing import Any, Generator
from weakref import WeakKeyDictionary

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tqdm import tqdm


class OverrideEqMode(TorchDispatchMode):
    """
    When using a torch.Tensor as a key in a dictionary, the equality
    check must return a single value instead of a torch.Tensor
    of bool values.
    """

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.eq.Tensor:
            assert len(args) == 2, "Exactly 2 args must be provided"
            return torch.tensor(id(args[0]) == id(args[1]))
        return func(*args, **kwargs)


offload_values: WeakKeyDictionary[torch.Tensor, torch.Tensor] = WeakKeyDictionary()


class IntermediatesCache:
    """
    Transparent proxy for automatic tensor offloading/onloading.
    
    Wraps any data structure (Tensor, list, dict, nested structures) and:
    - When offloaded: automatically moves tensors to offload_device on assignment
    - When accessed: automatically moves tensors to onload_device
    
    Supports: Tensor, list, dict, tuple, dataclass, and nested combinations.
    
    Usage:
        # Wrap existing data
        cache = IntermediatesCache(data, offload_device='cpu', onload_device='cuda')
        cache.offload()  # Move all tensors to CPU
        
        # Access auto-onloads
        tensor = cache['key']  # Tensor moved to cuda
        
        # Assignment auto-offloads (when offloaded)
        cache['key'] = gpu_tensor  # Tensor moved to cpu
        
        # Create empty container
        cache = IntermediatesCache([], offload_device='cpu')
        cache.append(gpu_tensor)  # Auto-offloads
    """

    def __init__(
        self,
        data: Any = None,
        offload_device: torch.device | str | None = "cpu",
        onload_device: torch.device | str | None = None,
        parent=None,
        key=None,
    ):
        self._data = data
        self._offload_device = torch.device(offload_device) if offload_device else None
        self._onload_device = torch.device(onload_device) if onload_device else None
        self._parent = parent
        self._key = key
        self._is_offloaded = offload_device is not None

    @classmethod
    def _offload_value(cls, value: Any, offload_device: torch.device | None, onload_device: torch.device | None = None) -> Any:
        """Recursively offload tensors to the offload device."""
        if offload_device is None:
            return value
        
        match value:
            case torch.Tensor():
                with OverrideEqMode():
                    if value in offload_values:
                        offloaded = offload_values[value]
                    else:
                        offloaded = value.to(device=offload_device)
                        if offloaded is not value:
                            if (
                                offload_device.type == "cpu"
                                and torch.accelerator.is_available()
                                and not offloaded.is_pinned()
                            ):
                                offloaded = offloaded.pin_memory()
                            offload_values[value] = offloaded
                return offloaded
            case list():
                return [cls._offload_value(v, offload_device, onload_device) for v in value]
            case tuple():
                return tuple(cls._offload_value(v, offload_device, onload_device) for v in value)
            case dict():
                return {k: cls._offload_value(v, offload_device, onload_device) for k, v in value.items()}
            case _ if is_dataclass(value):
                for field in fields(value):
                    v = getattr(value, field.name)
                    setattr(value, field.name, cls._offload_value(v, offload_device, onload_device))
                return value
            case _:
                if not isinstance(value, (int, str, float, bool, torch.dtype, torch.device, type(None))):
                    warnings.warn(f"Offloading not implemented for type {type(value)}.")
                return value

    @classmethod
    def _onload_value(cls, value: Any, onload_device: torch.device | None) -> Any:
        """Recursively onload tensors to the target device."""
        if onload_device is None:
            return value
        
        match value:
            case torch.Tensor():
                non_blocking = (
                    value.is_pinned()
                    and torch.accelerator.is_available()
                    and onload_device.type == torch.accelerator.current_accelerator().type
                )
                return value.to(device=onload_device, non_blocking=non_blocking)
            case list():
                return [cls._onload_value(v, onload_device) for v in value]
            case tuple():
                return tuple(cls._onload_value(v, onload_device) for v in value)
            case dict():
                return {k: cls._onload_value(v, onload_device) for k, v in value.items()}
            case _ if is_dataclass(value):
                for field in fields(value):
                    v = getattr(value, field.name)
                    setattr(value, field.name, cls._onload_value(v, onload_device))
                return value
            case _:
                return value

    def _process_input(self, value: Any) -> Any:
        """Process value for offloading if proxy is offloaded."""
        if not self._is_offloaded:
            return value
        return self._offload_value(value, self._offload_device, self._onload_device)

    def _process_output(self, value: Any) -> Any:
        """Process value for onloading if proxy is offloaded."""
        if not self._is_offloaded:
            return value
        return self._onload_value(value, self._onload_device)

    def __getitem__(self, key):
        child_data = self._data[key]
        if isinstance(child_data, IntermediatesCache):
            return child_data
        return IntermediatesCache(
            child_data,
            offload_device=self._offload_device,
            onload_device=self._onload_device,
            parent=self,
            key=key,
        )

    def __setitem__(self, key, value):
        processed = self._process_input(value)
        self._data[key] = processed
        if self._parent is not None:
            self._parent._data[self._key] = self._data

    def __delitem__(self, key):
        del self._data[key]
        if self._parent is not None:
            self._parent._data[self._key] = self._data

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IntermediatesCache(offloaded={self._is_offloaded}, data={type(self._data).__name__})"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._data, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                processed_args = [self._process_input(a) for a in args]
                processed_kwargs = {k: self._process_input(v) for k, v in kwargs.items()}
                result = attr(*processed_args, **processed_kwargs)
                if self._parent is not None:
                    self._parent._data[self._key] = self._data
                return result
            return wrapper
        return attr

    def unwrap(self) -> Any:
        """Get underlying data with tensors onloaded."""
        return self._process_output(self._data)

    def offload(self):
        """Offload all tensors to offload_device."""
        self._is_offloaded = True
        
        if isinstance(self._data, torch.Tensor):
            new_tensor = self._offload_value(self._data, self._offload_device, self._onload_device)
            self._data = new_tensor
            if self._parent is not None:
                self._parent._data[self._key] = new_tensor
        elif isinstance(self._data, (list, dict)):
            keys = range(len(self._data)) if isinstance(self._data, list) else list(self._data.keys())
            for k in keys:
                self[k].offload()
        elif isinstance(self._data, tuple):
            new_list = [self._offload_value(v, self._offload_device, self._onload_device) for v in self._data]
            self._data = tuple(new_list)
            if self._parent is not None:
                self._parent._data[self._key] = self._data
        elif is_dataclass(self._data):
            for field in fields(self._data):
                v = getattr(self._data, field.name)
                setattr(self._data, field.name, self._offload_value(v, self._offload_device, self._onload_device))

    def onload(self, device=None):
        """Onload all tensors to target device."""
        target_device = device or self._onload_device
        self._is_offloaded = False
        
        if isinstance(self._data, torch.Tensor):
            new_tensor = self._onload_value(self._data, target_device)
            self._data = new_tensor
            if self._parent is not None:
                self._parent._data[self._key] = new_tensor
        elif isinstance(self._data, (list, dict)):
            keys = range(len(self._data)) if isinstance(self._data, list) else list(self._data.keys())
            for k in keys:
                self[k].onload(device)
        elif isinstance(self._data, tuple):
            new_list = [self._onload_value(v, target_device) for v in self._data]
            self._data = tuple(new_list)
            if self._parent is not None:
                self._parent._data[self._key] = self._data
        elif is_dataclass(self._data):
            for field in fields(self._data):
                v = getattr(self._data, field.name)
                setattr(self._data, field.name, self._onload_value(v, target_device))

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        model_device: torch.device = torch.device("cpu"),
        offload_device: torch.device | None = torch.device("cpu"),
    ):
        """
        Create a cache from dataloader - returns list of batch dicts wrapped in proxy.
        
        :param dataloader: dataloader which generates values to be cached
        :param model_device: device which values will be onloaded to when fetched
        :param offload_device: device to offload values to
        """
        batch_intermediates = [
            cls._offload_value(batch, offload_device, model_device)
            for batch in tqdm(dataloader, desc="Preparing cache")
        ]
        return cls(batch_intermediates, offload_device, model_device)

    def select(self, keys: list[str]) -> dict:
        """
        Select specified keys from dict data, returning onloaded values.
        
        :param keys: list of keys to select
        :return: dict with onloaded values for selected keys
        """
        if not isinstance(self._data, dict):
            raise TypeError("select() only works when data is a dict")
        return {k: self[k].unwrap() for k in keys if k in self._data}

    def iter(self) -> Generator[IntermediatesCache, None, None]:
        """Iterate over list items, yielding each as a proxy."""
        if not isinstance(self._data, list):
            raise TypeError("iter() only works when data is a list")
        
        for i in range(len(self._data)):
            yield self[i]

    def iter_prefetch(self) -> Generator[IntermediatesCache, None, None]:
        """Iterate with background prefetch, yielding proxies with tensors onloaded."""
        if not isinstance(self._data, list):
            raise TypeError("iter_prefetch() only works when data is a list")
        
        num_batches = len(self._data)
        if num_batches == 0:
            return

        h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def _fetch(batch_index):
            event = None
            batch_proxy = self[batch_index]
            if h2d_stream is not None:
                with torch.cuda.stream(h2d_stream):
                    batch_proxy.onload()
                event = torch.cuda.Event()
                event.record(h2d_stream)
            else:
                batch_proxy.onload()
            return batch_proxy, event

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            for batch_index in range(num_batches):
                if future is not None:
                    current, event = future.result()
                else:
                    current, event = _fetch(batch_index)
                if batch_index + 1 < num_batches:
                    future = executor.submit(_fetch, batch_index + 1)
                else:
                    future = None
                if event is not None:
                    torch.cuda.current_stream().wait_event(event)
                yield current