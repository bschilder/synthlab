"""
SynthLab utility functions.

This module provides general-purpose utilities for working with synthetic healthcare data.
"""

from typing import Optional


def count_tokens(
    text: str | list[str],
    approximate: bool = False,
    model: str = "gpt-4o-mini",
    sep: str = " ",
) -> int:
    """
    Count tokens for text, optionally using a fast approximate estimate.

    Args:
        text: Text or list of text chunks.
        approximate: If True, uses a rough estimate (1 token ≈ 4 chars).
                    This is much faster but less accurate.
        model: Model name for tokenizer lookup when approximate=False.
               Default is "gpt-4o-mini" which uses cl100k_base encoding.
        sep: Join separator if text is a list.

    Returns:
        Number of tokens in the text.

    Example:
        >>> import synthlab as sl
        >>> sl.utils.count_tokens("Hello, world!")
        4
        >>> sl.utils.count_tokens("Hello, world!", approximate=True)
        3
    """
    if isinstance(text, list):
        text = sep.join(text)
    if approximate:
        return len(text) // 4

    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except ImportError:
        # Fall back to approximate if tiktoken not installed
        return len(text) // 4


def estimate_tokens(text: str | list[str], sep: str = " ") -> int:
    """
    Fast approximate token count (1 token ≈ 4 characters).

    This is much faster than count_tokens() but less accurate.
    Use for quick estimates when exact counts aren't needed.

    Args:
        text: Text or list of text chunks.
        sep: Join separator if text is a list.

    Returns:
        Approximate number of tokens.

    Example:
        >>> import synthlab as sl
        >>> sl.utils.estimate_tokens("Hello, world!")
        3
    """
    return count_tokens(text, approximate=True, sep=sep)


def format_size(size_bytes: int) -> str:
    """
    Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 GB", "256 MB").

    Example:
        >>> import synthlab as sl
        >>> sl.utils.format_size(1024 * 1024 * 1024)
        '1.0 GB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def check_gpu_availability() -> dict:
    """
    Check GPU availability and memory.

    Returns:
        Dictionary with GPU information:
        - available: bool - whether CUDA is available
        - count: int - number of GPUs
        - devices: list of dicts with name, memory_total, memory_free for each GPU

    Example:
        >>> import synthlab as sl
        >>> info = sl.utils.check_gpu_availability()
        >>> print(f"GPUs available: {info['count']}")
    """
    result = {
        "available": False,
        "count": 0,
        "devices": [],
    }

    try:
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["count"] = torch.cuda.device_count()

            for i in range(result["count"]):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory
                mem_free = mem_total - torch.cuda.memory_allocated(i)

                result["devices"].append({
                    "id": i,
                    "name": props.name,
                    "memory_total": mem_total,
                    "memory_total_str": format_size(mem_total),
                    "memory_free": mem_free,
                    "memory_free_str": format_size(mem_free),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass

    return result


def print_gpu_info():
    """
    Print GPU availability and memory information.

    Example:
        >>> import synthlab as sl
        >>> sl.utils.print_gpu_info()
        GPU Information
        ===============
        CUDA available: Yes
        GPU count: 2

        GPU 0: NVIDIA A100-SXM4-40GB
          Memory: 40.0 GB total, 38.5 GB free
          Compute: 8.0
    """
    info = check_gpu_availability()

    print("GPU Information")
    print("=" * 40)
    print(f"CUDA available: {'Yes' if info['available'] else 'No'}")

    if info['available']:
        print(f"GPU count: {info['count']}")
        print()

        for gpu in info['devices']:
            print(f"GPU {gpu['id']}: {gpu['name']}")
            print(f"  Memory: {gpu['memory_total_str']} total, {gpu['memory_free_str']} free")
            print(f"  Compute: {gpu['compute_capability']}")
    else:
        print("No CUDA GPUs detected. Install PyTorch with CUDA support for GPU acceleration.")


__all__ = [
    "count_tokens",
    "estimate_tokens",
    "format_size",
    "check_gpu_availability",
    "print_gpu_info",
]
