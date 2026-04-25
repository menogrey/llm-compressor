from dataclasses import dataclass, fields, is_dataclass

import pytest
import torch
from torch.utils.data import DataLoader, StackDataset

from llmcompressor.pipelines.cache import IntermediatesCache, OverrideEqMode


@dataclass
class SampleDataclass:
    a: torch.Tensor
    b: int


@pytest.fixture
def sample_dataloader():
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    dataset = StackDataset(input_ids=input_ids, attention_mask=attention_mask)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def sample_cache(sample_dataloader):
    return IntermediatesCache.from_dataloader(
        dataloader=sample_dataloader,
        model_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


values_to_test = [
    torch.randn(2, 3).to("cpu"),
    SampleDataclass(a=torch.randn(2, 3), b=42),
    torch.float32,
    [1, 2, 3],
]


@pytest.mark.unit
def test_initialization(sample_dataloader):
    cache = IntermediatesCache.from_dataloader(
        dataloader=sample_dataloader,
        model_device=torch.device("cpu"),
    )

    assert isinstance(cache, IntermediatesCache)
    assert len(cache) > 0
    assert isinstance(cache._data, list)
    assert isinstance(cache._data[0], dict)


@pytest.mark.unit
def test_iter_empty_cache():
    cache = IntermediatesCache([], torch.device("cpu"))
    assert list(cache.iter()) == []


@pytest.mark.unit
def test_iter_yields_proxies(sample_cache):
    for batch_proxy in sample_cache.iter():
        assert isinstance(batch_proxy, IntermediatesCache)
        batch = batch_proxy.unwrap()
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert isinstance(batch["input_ids"], torch.Tensor)


@pytest.mark.unit
def test_iter_prefetch_yields_proxies(sample_cache):
    for batch_proxy in sample_cache.iter_prefetch():
        assert isinstance(batch_proxy, IntermediatesCache)
        batch = batch_proxy.unwrap()
        assert isinstance(batch, dict)


@pytest.mark.unit
def test_iter_prefetch_matches_iter(sample_cache):
    def batch_dicts_equal(a: dict, b: dict) -> bool:
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    via_iter = [p.unwrap() for p in sample_cache.iter()]
    via_prefetch = [p.unwrap() for p in sample_cache.iter_prefetch()]
    assert len(via_iter) == len(via_prefetch)
    for i, (b_iter, b_prefetch) in enumerate(zip(via_iter, via_prefetch)):
        assert batch_dicts_equal(b_iter, b_prefetch), f"batch {i} differs"


@pytest.mark.unit
def test_select_keys(sample_cache):
    batch_proxy = sample_cache[0]
    selected = batch_proxy.select(["input_ids"])
    
    assert isinstance(selected, dict)
    assert "input_ids" in selected
    assert "attention_mask" not in selected
    assert isinstance(selected["input_ids"], torch.Tensor)


@pytest.mark.unit
def test_fetch_inputs(sample_cache):
    batch = sample_cache[0].unwrap()

    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)


@pytest.mark.unit
def test_update_intermediates(sample_cache):
    new_outputs = {
        "hidden_states": torch.randn(2, 4, 768),
        "logits": torch.randn(2, 4, 1000),
    }

    sample_cache._data[0].update(
        IntermediatesCache._offload_value(new_outputs, sample_cache._offload_device, sample_cache._onload_device)
    )

    assert "hidden_states" in sample_cache._data[0]
    assert "logits" in sample_cache._data[0]


@pytest.mark.unit
def test_delete_intermediates(sample_cache):
    new_outputs = {
        "hidden_states": torch.randn(2, 4, 768),
        "logits": torch.randn(2, 4, 1000),
    }
    sample_cache._data[0].update(
        IntermediatesCache._offload_value(new_outputs, sample_cache._offload_device, sample_cache._onload_device)
    )

    del sample_cache._data[0]["hidden_states"]

    assert "hidden_states" not in sample_cache._data[0]
    assert "logits" in sample_cache._data[0]


@pytest.mark.unit
@pytest.mark.parametrize("value", values_to_test)
def test_from_dataloader(value):
    dataset = StackDataset(value=[value])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    cache = IntermediatesCache.from_dataloader(dataloader)

    onloaded = cache[0]["value"].unwrap()
    assert deep_equal(onloaded, value)


@pytest.mark.unit
@pytest.mark.parametrize("value", values_to_test)
def test_offload_and_onload(value):
    offloaded = IntermediatesCache._offload_value(value, torch.device("cpu"))
    onloaded = IntermediatesCache._onload_value(offloaded, torch.device("cpu"))
    assert deep_equal(onloaded, value)


@pytest.mark.unit
def test_device_handling(sample_dataloader):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cuda_device = torch.device("cuda")
    cpu_device = torch.device("cpu")

    cache = IntermediatesCache.from_dataloader(
        dataloader=sample_dataloader,
        model_device=cuda_device,
        offload_device=cpu_device,
    )

    new_outputs = {"hidden_states": torch.randn(2, 3).to(cuda_device)}
    cache._data[0].update(
        IntermediatesCache._offload_value(new_outputs, cpu_device, cuda_device)
    )

    assert cache._data[0]["hidden_states"].device.type == "cpu"

    hidden = cache[0]["hidden_states"].unwrap()
    assert hidden.device.type == "cuda"


@pytest.mark.unit
def test_transparent_list_append():
    cache = IntermediatesCache([], offload_device="cpu", onload_device="cuda")
    
    tensor = torch.randn(2, 3)
    cache.append(tensor)
    
    assert len(cache) == 1
    assert cache._data[0].device.type == "cpu"


@pytest.mark.unit
def test_transparent_dict_assignment():
    cache = IntermediatesCache({}, offload_device="cpu", onload_device="cuda")
    
    tensor = torch.randn(2, 3)
    cache["key"] = tensor
    
    assert "key" in cache._data
    assert cache._data["key"].device.type == "cpu"


@pytest.mark.unit
def test_transparent_access_onload():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    cache = IntermediatesCache(
        {"key": torch.randn(2, 3)},
        offload_device="cpu",
        onload_device="cuda",
    )
    cache.offload()
    
    assert cache._data["key"].device.type == "cpu"
    
    unwrapped = cache.unwrap()
    assert unwrapped["key"].device.type == "cuda"


@pytest.mark.unit
def test_nested_structure():
    data = {
        "outer": [
            {"inner": torch.randn(2, 3)},
            torch.randn(3, 4),
        ]
    }
    
    cache = IntermediatesCache(data, offload_device="cpu", onload_device="cuda")
    cache.offload()
    
    assert cache._data["outer"][0]["inner"].device.type == "cpu"
    assert cache._data["outer"][1].device.type == "cpu"


def deep_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False

    match a:
        case torch.Tensor():
            return torch.equal(a, b)
        case list() | tuple():
            if len(a) != len(b):
                return False
            return all(deep_equal(_a, _b) for _a, _b in zip(a, b))
        case dict():
            if a.keys() != b.keys():
                return False
            return all(deep_equal(a[key], b[key]) for key in a.keys())
        case _ if is_dataclass(a):
            a_dict = {field.name: getattr(a, field.name) for field in fields(a)}
            b_dict = {field.name: getattr(b, field.name) for field in fields(b)}
            return deep_equal(a_dict, b_dict)
        case _:
            return a == b


def test_override_eq_mode():
    a = torch.tensor([1, 2, 3])
    b = a
    c = torch.tensor([2, 2, 2])

    with pytest.raises(RuntimeError):
        assert a == b
    with pytest.raises(RuntimeError):
        assert not (a == c)

    with OverrideEqMode():
        assert a == b
        assert not (a == c)