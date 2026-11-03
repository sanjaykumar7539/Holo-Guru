"""Print conservative runtime recommendations for the current machine."""
import os


def main():
    cpu = os.cpu_count() or 1
    threads = max(1, min(8, cpu // 2))
    print(f"CPU cores detected: {cpu}")
    print(f"Recommended OMP_NUM_THREADS={threads}")
    print(f"Recommended MKL_NUM_THREADS={threads}")
    print("Keep LLM_MAX_CONCURRENT=1 on memory-constrained single-device deployments.")
    print("Use OpenVINO for embedding/reranker on Intel CPU; keep CUDA/4-bit path for Qwen when supported.")
    print("Do not enable image/video generation on the latency-critical classroom path.")


if __name__ == "__main__":
    main()
