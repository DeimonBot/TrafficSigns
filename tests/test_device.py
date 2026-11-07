from __future__ import annotations

import pytest
import torch

from roadvision.device import elegir_dispositivo


def test_auto_sin_cuda_usa_cpu_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = elegir_dispositivo("auto", fp16=True)
    assert (d.nombre, d.dtype, d.descripcion) == ("cpu", torch.float32, "CPU")


def test_auto_con_cuda_usa_fp16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "GPU de prueba")
    d = elegir_dispositivo("auto", fp16=True)
    assert (d.nombre, d.dtype, d.descripcion) == ("cuda", torch.float16, "GPU de prueba")
    assert elegir_dispositivo("auto", fp16=False).dtype == torch.float32


def test_cuda_pedido_sin_gpu_falla(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        elegir_dispositivo("cuda")


def test_valor_invalido():
    with pytest.raises(ValueError):
        elegir_dispositivo("tpu")
