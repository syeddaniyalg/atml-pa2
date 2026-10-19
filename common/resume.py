from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import torch

from common.data import repo_path


def trainable_state(model) -> dict:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(directory, models: dict, optimizers: dict, meta: dict) -> None:
    final = repo_path(directory)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, model in models.items():
        torch.save(trainable_state(model), tmp / f"{name}.pt")
    for name, optimizer in optimizers.items():
        torch.save(optimizer.state_dict(), tmp / f"{name}_optimizer.pt")
    (tmp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


def load_checkpoint(directory, models: dict, optimizers: dict):
    final = repo_path(directory)
    if not (final / "meta.json").exists():
        return None
    for name, model in models.items():
        saved = torch.load(final / f"{name}.pt", map_location="cpu")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for key, value in saved.items():
                params[key].copy_(value.to(device=params[key].device, dtype=params[key].dtype))
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(torch.load(final / f"{name}_optimizer.pt", map_location="cpu"))
    return json.loads((final / "meta.json").read_text(encoding="utf-8"))


def clear_checkpoint(directory) -> None:
    final = repo_path(directory)
    if final.exists():
        shutil.rmtree(final)


def archive_file(path):
    p = repo_path(path)
    if not p.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = p.with_name(f"{p.stem}.{stamp}{p.suffix}.bak")
    os.replace(p, target)
    return target


def archive_dir(path):
    p = repo_path(path)
    if not p.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = p.with_name(f"{p.name}.{stamp}.bak")
    os.replace(p, target)
    return target


def read_log(path) -> list[dict]:
    p = repo_path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_log(path, rows: list[dict]) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def prepare_log(path, key: str, limit) -> list[dict]:
    if limit is None:
        archive_file(path)
        return []
    rows = [row for row in read_log(path) if row.get(key, 0) <= limit]
    write_log(path, rows)
    return rows


def save_json_atomic(path, obj) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def completed_summary(summary_path, adapter_dir):
    p = repo_path(summary_path)
    if p.exists() and (repo_path(adapter_dir) / "adapter_config.json").exists():
        summary = json.loads(p.read_text(encoding="utf-8"))
        if summary.get("completed"):
            return summary
    return None


def read_partial(path) -> list[dict]:
    return read_log(path)


def append_partial(path, rows: list[dict]) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_file(path) -> None:
    p = Path(repo_path(path))
    if p.exists():
        p.unlink()
