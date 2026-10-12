from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# A git checkout of the patched runtime (external/sglang) wins when present. The standalone package instead ships the files the patch
# series modifies (patched_sglang_files/) plus four unmodified upstream files some tests read (upstream_support_files/); the two are
# merged into one temporary tree so every test sees a single `srt` root.
def _runtime_root() -> Path:
    import atexit
    import shutil
    import tempfile

    checkout = PROJECT_ROOT / "external" / "sglang" / "python" / "sglang" / "srt"
    if (checkout / "pic_runtime.py").is_file():
        return checkout
    patched = PROJECT_ROOT / "patched_sglang_files" / "python" / "sglang" / "srt"
    if not (patched / "pic_runtime.py").is_file():
        return checkout
    merged = Path(tempfile.mkdtemp(prefix="noc_srt_view_")) / "python" / "sglang" / "srt"
    atexit.register(shutil.rmtree, merged.parents[2], ignore_errors=True)
    for source in (PROJECT_ROOT / "upstream_support_files" / "python" / "sglang" / "srt", patched):  # patched files win
        if source.is_dir():
            shutil.copytree(source, merged, dirs_exist_ok=True)
    return merged


SRT_ROOT = _runtime_root()
SGLANG_PYTHON = SRT_ROOT.parents[1]


def load_srt_module(name: str, filename: str):
    module_name = f"pic_hardware_independent_tests.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SRT_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeRotary:
    def __init__(
        self,
        *,
        max_position: int = 128,
        rotary_dim: int = 8,
        dtype: torch.dtype = torch.float32,
        theta: float = 10_000.0,
    ) -> None:
        self.rotary_dim = rotary_dim
        self.is_neox_style = True
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )
        positions = torch.arange(max_position, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
            dtype
        )

    def _ensure_cos_sin_cache_length(self, needed_max_pos: int) -> None:
        if needed_max_pos >= self.cos_sin_cache.shape[0]:
            raise AssertionError("fixture RoPE cache is too short")


def make_pic_runtime(rotary: FakeRotary):
    module = load_srt_module("pic_runtime", "pic_runtime.py")
    runtime = module.PICRuntime.__new__(module.PICRuntime)
    runtime._rotary_by_layer = {0: rotary}
    return module, runtime


def fixed_generator(seed: int = 20260824) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)

