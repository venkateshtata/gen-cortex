import ray
import torch

ray.init()

@ray.remote(num_gpus=1)  # Ensure Ray assigns a GPU to this task
def check_gpu():
    try:
        available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device() if available else "No GPU"
        device_name = torch.cuda.get_device_name(current_device) if available else "N/A"
        return {
            "gpu_available": available,
            "device_count": device_count,
            "current_device": current_device,
            "device_name": device_name,
        }
    except Exception as e:
        return {"error": str(e)}

result = ray.get(check_gpu.remote())
print(result)
