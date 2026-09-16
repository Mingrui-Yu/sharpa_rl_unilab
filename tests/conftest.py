import pytest
import torch


@pytest.fixture(autouse=True)
def small_torch_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            yield
    finally:
        torch.set_num_threads(previous)
