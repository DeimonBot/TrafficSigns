"""Verifica que PyTorch ve una GPU NVIDIA con CUDA y que fp16 funciona."""
from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"CUDA (build): {torch.version.cuda}")
    hay_cuda = torch.cuda.is_available()
    print(f"CUDA disponible: {hay_cuda}")
    if not hay_cuda:
        print("No se detectó una GPU NVIDIA con CUDA")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {props.total_memory / 1024**3:.1f} GB")
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"matmul fp16 en GPU: OK {tuple(c.shape)}, finito={bool(torch.isfinite(c).all())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
