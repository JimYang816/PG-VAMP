"""Bound small algebra tests to the same reproducible CPU thread budget."""

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(4)
