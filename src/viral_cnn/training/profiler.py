"""
Resource profiling and telemetry for controlled architecture training.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import time
from typing import Any, Iterator, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ResourceProfile:
    """
    Standardized telemetry output documenting computational resource consumption.
    """

    device_name: str
    peak_vram_mb: float
    mean_step_time_ms: float
    throughput_seq_per_sec: float
    total_flops: float
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceProfiler:
    """
    Context manager and tracker capturing step latencies, memory footprint, and throughput.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.step_times: list[float] = []
        self.batch_sizes: list[int] = []
        self.start_wall_time = time.perf_counter()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def profile_step(self, batch_size: int) -> Iterator[None]:
        """
        Profiles a single forward-backward-update step.
        """
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        t0 = time.perf_counter()
        yield
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()

        self.step_times.append(t1 - t0)
        self.batch_sizes.append(batch_size)

    def estimate_model_flops(
        self, model: nn.Module, input_shape: tuple[int, ...] = (1, 4, 512)
    ) -> float:
        """
        Estimates total FLOPs per batch sample based on parameter operations.
        """
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Standard heuristic for 1D CNNs: approx 2 MACs (4 FLOPs) per parameter per sequence length factor
        seq_len = input_shape[-1] if len(input_shape) >= 3 else 512
        flops_per_sample = float(total_params * (seq_len / 256.0) * 4.0)
        return max(flops_per_sample, 1.0e6)

    def finalize(
        self,
        total_steps: int | None = None,
        model: nn.Module | None = None,
        input_shape: tuple[int, ...] = (1, 4, 512),
    ) -> ResourceProfile:
        """
        Compiles all collected step metrics into a ResourceProfile.
        """
        wall_clock = time.perf_counter() - self.start_wall_time
        steps = (
            total_steps
            if total_steps is not None and total_steps > 0
            else len(self.step_times)
        )

        if not self.step_times:
            mean_step_ms = 0.0
            throughput = 0.0
        else:
            total_step_time = sum(self.step_times)
            mean_step_ms = (total_step_time / len(self.step_times)) * 1000.0
            total_sequences = sum(self.batch_sizes)
            throughput = (
                total_sequences / total_step_time if total_step_time > 0 else 0.0
            )

        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (
                1024.0 * 1024.0
            )
            device_name = torch.cuda.get_device_name(self.device)
        else:
            # Fallback for CPU
            peak_vram = 0.0
            device_name = "CPU"

        flops_per_sample = (
            self.estimate_model_flops(model, input_shape) if model is not None else 1.0e9
        )
        total_samples = sum(self.batch_sizes) if self.batch_sizes else steps * 32
        total_flops = float(flops_per_sample * total_samples)

        return ResourceProfile(
            device_name=device_name,
            peak_vram_mb=float(peak_vram),
            mean_step_time_ms=float(mean_step_ms),
            throughput_seq_per_sec=float(throughput),
            total_flops=float(total_flops),
            wall_clock_seconds=float(wall_clock),
        )


def profile_inference_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (32, 512),
    device: Union[str, torch.device] = "cpu",
    num_warmup: int = 5,
    num_runs: int = 20,
) -> dict[str, Any]:
    """
    Profiles an inference model to measure parameters, memory footprint,
    latency, throughput, and estimated GFLOPs.
    Matches the historical benchmark.ipynb profile_model contract.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)
    model.eval()

    if len(input_size) == 2:
        x = torch.randint(0, 4, input_size, dtype=torch.long, device=dev)
    else:
        x = torch.randn(input_size, dtype=torch.float32, device=dev)

    # 1. Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    head_params = 0
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        head_params = sum(p.numel() for p in model.fc.parameters())
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        head_params = sum(p.numel() for p in model.classifier.parameters())
    backbone_params = int(total_params - head_params)

    # 2. Footprint in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024.0 ** 2)

    # 3. FLOPs estimation per sample
    seq_len = input_size[-1]
    flops_per_sample = float(total_params * (seq_len / 256.0) * 4.0)
    gflops = round(flops_per_sample / 1.0e9, 4)

    # 4. Warmup passes
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(dev)

    # 5. Latency benchmarking
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = model(x)
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    std_latency = float(np.std(latencies)) if latencies else 0.0

    batch_size = input_size[0]
    throughput = (batch_size / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0

    return {
        "backbone_params": int(backbone_params),
        "head_params": int(head_params),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "size_mb": round(float(total_size_mb), 2),
        "gflops": gflops,
        "avg_latency_ms": round(avg_latency, 2),
        "std_latency_ms": round(std_latency, 2),
        "throughput_seq_sec": round(float(throughput), 1),
    }
