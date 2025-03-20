import os
import torch
from dotenv import load_dotenv
from pathlib import Path


def path_check(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return Path(dir)


def load_env():
    env_file = Path(os.getcwd()).joinpath("sample.env")
    load_dotenv(env_file)

    return os.getenv("HUGGINGFACE_AUTH_TOKEN")


def set_device(device: str):
    if device is None:
        print("Setting device='cpu'")
        return "cpu"
    elif device == "auto":
        print("Setting device='auto'")
        return device
    try:
        num = int(device)
        print(f"Setting device='cuda:{num}'")
        if torch.cuda.is_available():
            if torch.cuda.device_count() > num:
                return f"cuda:{num}"
            else:
                raise ValueError(f"Device {num} of cuda is not available.")
        else:
            raise ValueError("Cuda device is not available.")
    except:
        raise ValueError(f"Unrecognized cuda device number: {device}")
