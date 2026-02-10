import torch
import gc


def clear_gradients(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.grad is not None:
            arg.grad = None


def clear_memory(device):
    if isinstance(device, str):
        device_type = device.split(":")[0]
    else:
        device_type = device.type if hasattr(device, "type") else str(device)
    if device_type == "cuda":
        torch._C._cuda_clearCublasWorkspaces()
    torch._dynamo.reset()
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device_type == "xpu":
        torch.xpu.empty_cache()


def peak_memory(f, *args, device):
    if isinstance(device, str):
        device_type = device.split(":")[0]
    else:
        device_type = device.type if hasattr(device, "type") else str(device)
    for _ in range(3):
        # Clean everything
        clear_memory(device)
        clear_gradients(*args)

        # Run once
        f(*args)

        # Measure peak memory
        if device_type == "cuda":
            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated(device)
        elif device_type == "xpu":
            torch.xpu.synchronize()
            memory = torch.xpu.max_memory_allocated(device)
        else:
            memory = 0

    return memory


def current_memory(device):
    if isinstance(device, str):
        device_type = device.split(":")[0]
    else:
        device_type = device.type if hasattr(device, "type") else str(device)
    if device_type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024**3)
    elif device_type == "xpu":
        return torch.xpu.memory_allocated(device) / (1024**3)
    return 0.0


def memory_measure(f, device, num_iters=3):
    # Clean everything
    clear_memory(device)

    # Run measurement
    print("Current memory: ", current_memory(device))
    memory = peak_memory(f, device=device)

    print("Peak memory: ", memory / (1024**3))
    return memory / (1024**3)


def memory_measure_simple(f, device, *args, **kwargs):
    if isinstance(device, str):
        device_type = device.split(":")[0]
    else:
        device_type = device.type if hasattr(device, "type") else str(device)
    # Clean everything
    clear_memory(device)
    clear_gradients(*args)

    current = current_memory(device)

    # Run once
    out = f(*args, **kwargs)

    # Measure peak memory
    if device_type == "cuda":
        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated(device)
    elif device_type == "xpu":
        torch.xpu.synchronize()
        memory = torch.xpu.max_memory_allocated(device)
    else:
        memory = 0
    memory = memory / (1024**3)
    memory = memory - current

    return out, memory
