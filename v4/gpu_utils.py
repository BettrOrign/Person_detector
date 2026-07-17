import logging

logger = logging.getLogger(__name__)

_HAS_CUDA = None


def has_cuda() -> bool:
    global _HAS_CUDA
    if _HAS_CUDA is not None:
        return _HAS_CUDA
    try:
        import torch
        _HAS_CUDA = torch.cuda.is_available()
    except ImportError:
        _HAS_CUDA = False
    if _HAS_CUDA:
        logger.info("CUDA detected")
    else:
        logger.info("CUDA not available, using CPU")
    return _HAS_CUDA


def onnx_providers() -> list[str]:
    if has_cuda():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def torch_device() -> str:
    return "cuda:0" if has_cuda() else "cpu"
