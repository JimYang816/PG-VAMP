"""Deterministic no-grad replay and bounded, mutation-safe matrix caching."""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from ..channel.affine import receive_window
from ..channel.effective_matrix import effective_matrix
from ..channel.noise import complex_awgn, sigma2_from_esn0
from ..modulation.qpsk import bits_to_symbols
from ..receiver.fft_receiver import fft_receive
from ..receiver.preprocessing import preprocess
from ..utils.device import resolve_runtime
from ..utils.random import generator, stable_hash
from .manifest import load_manifest
from .records import MODEL_VERSION, record_frame, record_paths, tensor_hash


def detection_inputs(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    """The only ordinary detector projection: no IDs, SNR class, paths or targets."""
    return {key: sample[key].clone() for key in ("H", "y", "sigma2")}


class EffectiveDataset:
    """One item per OFDM block. Storage is CPU; device selection is explicit."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        *,
        device: str = "cpu",
        cache_entries: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.manifest, self.config, records = load_manifest(manifest)
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        self.split = split
        self.records = [r for r in records if r["split"] == split]
        self.blocks = self.config.values["frame"]["n_ofdm_symbols"]
        rt = self.config.values["runtime"]
        self.runtime = resolve_runtime(device, rt["dtype"], rt["cpu_threads"], rt["deterministic"])
        self.cache_entries = (
            self.config.values["data"]["matrix_cache_entries"]
            if cache_entries is None
            else cache_entries
        )
        if type(self.cache_entries) is not int or self.cache_entries < 0:
            raise ValueError("cache_entries must be nonnegative integer")
        if self.config.values["receiver"]["sync_mode"] != "oracle_timing":
            raise ValueError("compact replay requires oracle_timing")
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records) * self.blocks

    @torch.no_grad()
    def __getitem__(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("dataset sample index out of range")
        record, block = self.records[index // self.blocks], index % self.blocks
        frame = record_frame(record, self.config, self.runtime)
        paths = record_paths(record, self.runtime.device)
        # Include all persisted physical payload, actual symbol time, model, mapping and runtime.
        key = stable_hash(
            [
                tensor_hash(record),
                block,
                frame.layout.useful[block],
                MODEL_VERSION,
                self.manifest["allocation"],
                self.manifest["config_sha256"],
                str(self.runtime.device),
                str(self.runtime.complex_dtype),
            ]
        )
        if key in self._cache:
            h = self._cache.pop(key)
            self._cache[key] = h
        else:
            h = effective_matrix(
                paths, frame.layout, self.config, block, dtype=self.runtime.complex_dtype
            )
            if self.cache_entries:
                self._cache[key] = h
                while len(self._cache) > self.cache_entries:
                    self._cache.popitem(last=False)
        sigma2 = sigma2_from_esn0(float(record["esn0_db"][block]))
        noise = complex_awgn(
            (self.config.values["waveform"]["n_fft_wave"],),
            sigma2,
            gen_r=generator(int(record["noise_seed_real"][block]), self.runtime.device),
            gen_i=generator(int(record["noise_seed_imag"][block]), self.runtime.device),
            dtype=self.runtime.complex_dtype,
        )
        if self.config.values["data"]["backend"] == "waveform_reference":
            clean = fft_receive(
                receive_window(frame, paths, self.config, block)[0].to(self.runtime.complex_dtype),
                frame.allocation,
            )
        else:
            clean = h @ frame.grid[0, block]
        observed = clean + fft_receive(noise, frame.allocation)
        pilots = frame.grid[0, block, frame.allocation.pilot_grid_index]
        result = preprocess(h, observed, pilots, frame.allocation, sigma2)
        bits = record["data_bits"][block].to(self.runtime.device).clone()
        return {
            "sample_id": f"{record['frame_id']}:{record['snr_copy']}:{block}",
            "frame_id": record["frame_id"],
            "channel_id": record["channel_id"],
            "split": self.split,
            "block": block,
            "H": result.H.clone(),
            "y": result.y.clone(),
            "sigma2": torch.tensor(
                sigma2, dtype=self.runtime.real_dtype, device=self.runtime.device
            ),
            "x": bits_to_symbols(bits, dtype=self.runtime.complex_dtype),
            "bits": bits,
        }
