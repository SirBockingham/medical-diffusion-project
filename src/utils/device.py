import torch


def resolve_device(device_setting: str = "auto") -> torch.device:
    # ROCm builds of PyTorch also report AMD GPUs through torch.cuda
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device_setting)