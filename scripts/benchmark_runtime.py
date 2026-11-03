"""Lightweight local benchmark for HOLO-GURU core overhead.
Does not download models. Run after installing project dependencies.
"""
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except Exception as e:
        result["torch_error"] = str(e)

    from core import observability
    t0 = time.perf_counter()
    for _ in range(10000):
        observability.record("benchmark.noop", 0.01)
    result["observability_10k_record_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    out = ROOT / "data" / "runtime_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
