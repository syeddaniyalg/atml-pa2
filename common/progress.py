from __future__ import annotations

from tqdm import tqdm


def postfix(**values) -> dict:
    out = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out[key] = str(value)
        else:
            out[key] = f"{value:.4g}"
    return out


def bar(iterable=None, total=None, desc: str = "", initial: int = 0, unit: str = "it", leave: bool = True):
    return tqdm(iterable, total=total, desc=desc, initial=initial, unit=unit, leave=leave, dynamic_ncols=True)
