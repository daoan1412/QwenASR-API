"""Cho vLLM 0.14 chạy với CUDA_VISIBLE_DEVICES=MIG-<uuid> (GPU chia MIG 11/11/2026).

vLLM `Platform.device_id_to_physical_device_id` ép `int()` từng mục của CUDA_VISIBLE_DEVICES để lấy chỉ số NVML
→ UUID MIG báo ValueError. Bản vá: mục là MIG-… thì trả chỉ số NVML của GPU CHA (NVML không bị CUDA_VISIBLE_DEVICES
chi phối, chỉ dùng để đọc compute capability / tên / tổng VRAM). Còn CUDA (torch) vẫn chỉ thấy đúng lát MIG đó;
vLLM cấp VRAM theo torch.cuda.mem_get_info() nên tỉ lệ gpu_memory_utilization tính trên VRAM của LÁT.
Chỗ vLLM ghi đè CUDA_VISIBLE_DEVICES bằng hàm này (get_device_indices) chỉ chạy khi data parallel > 1 / Ray — ASR không dùng.

run.sh chỉ thêm thư mục này vào PYTHONPATH khi CUDA_VISIBLE_DEVICES bắt đầu bằng MIG-; tiến trình con spawn (EngineCore)
kế thừa PYTHONPATH nên cũng được vá. Vá trễ: chỉ khi module vllm.platforms.interface được nạp.
"""
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "vllm.platforms.interface"


def _patch(mod) -> None:
    orig = mod.Platform.device_id_to_physical_device_id.__func__
    cache: dict[str, int] = {}

    def device_id_to_physical_device_id(cls, device_id: int):
        raw = os.environ.get(cls.device_control_env_var, "")
        ids = raw.split(",") if raw else []
        if ids and ids[device_id].startswith("MIG-"):
            uuid = ids[device_id]
            if uuid not in cache:
                from vllm.utils.import_utils import import_pynvml

                nvml = import_pynvml()
                nvml.nvmlInit()
                try:
                    mig = nvml.nvmlDeviceGetHandleByUUID(uuid)
                    parent = nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(mig)
                    cache[uuid] = int(nvml.nvmlDeviceGetIndex(parent))
                finally:
                    nvml.nvmlShutdown()
            return cache[uuid]
        return orig(cls, device_id)

    mod.Platform.device_id_to_physical_device_id = classmethod(device_id_to_physical_device_id)


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_orig = spec.loader.exec_module

        def exec_module(module):
            exec_orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("MIG-"):
    sys.meta_path.insert(0, _Hook())
