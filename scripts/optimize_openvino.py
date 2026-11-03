"""Optional OpenVINO export helper for CPU/Intel deployments.

This is deliberately optional. It never changes the normal model path and
requires optimum-intel/openvino only when explicitly run.

Examples:
  python scripts/optimize_openvino.py --model BAAI/bge-m3 --task feature-extraction
  python scripts/optimize_openvino.py --model BAAI/bge-reranker-v2-m3 --task text-classification
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="feature-extraction")
    ap.add_argument("--output", default=None)
    ap.add_argument("--quantize", choices=["none", "int8"], default="int8")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    output = Path(args.output or f"models/openvino/{safe}")
    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "optimum.exporters.openvino",
        "--model", args.model,
        "--task", args.task,
        str(output),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)

    if args.quantize == "int8":
        print("Export complete. For INT8, use optimum-intel/NNCF calibration for the target model and dataset; no unsafe blind quantization is applied here.")
    print(f"OpenVINO model: {output}")


if __name__ == "__main__":
    main()
