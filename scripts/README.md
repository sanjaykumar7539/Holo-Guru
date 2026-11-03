# Edge / Performance Utilities

These scripts are optional and never replace the normal runtime automatically.

- `edge_profile.py` prints conservative thread/device guidance.
- `benchmark_runtime.py` records a lightweight local machine benchmark to `data/runtime_benchmark.json` without downloading models.
- `optimize_openvino.py` exports a Hugging Face model to OpenVINO when the optional `requirements-edge.txt` stack is installed.

Recommended first candidates for OpenVINO on Intel CPU are the embedding and reranker models. Keep Qwen on the existing CUDA/4-bit path when a supported GPU is available.
