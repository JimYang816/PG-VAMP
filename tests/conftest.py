"""Bound small algebra tests to the same reproducible CPU thread budget."""

import pytest
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(4)


@pytest.fixture
def data_config():
    from pgvamp_ofdm.config import load_config

    return load_config("configs/wp3_smoke.yaml")


@pytest.fixture(scope="session")
def data_manifest(tmp_path_factory):
    from pgvamp_ofdm.config import load_config
    from pgvamp_ofdm.data.generate import generate_dataset

    return generate_dataset(
        load_config("configs/wp3_smoke.yaml"), tmp_path_factory.mktemp("data") / "compact"
    )
