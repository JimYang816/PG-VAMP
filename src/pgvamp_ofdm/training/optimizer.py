"""Adam with CPU-safe graph-capture checks across supported PyTorch releases."""

import torch


class DeviceExplicitAdam(torch.optim.Adam):
    """Keep PyTorch Adam math/state; CPU updates must not query CUDA availability.

    PyTorch's graph-capture health check queries CUDA even for CPU parameters.
    Bypass only that irrelevant check for an entirely CPU optimizer; CUDA keeps
    the upstream checks. Both hook names cover the upstream rename.
    """

    def _cpu_only(self) -> bool:
        return all(p.device.type == "cpu" for g in self.param_groups for p in g["params"])

    def _accelerator_graph_capture_health_check(self) -> None:
        if not self._cpu_only():
            super()._accelerator_graph_capture_health_check()

    def _cuda_graph_capture_health_check(self) -> None:
        if not self._cpu_only():
            super()._cuda_graph_capture_health_check()
