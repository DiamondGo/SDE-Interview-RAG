"""Layered configuration: config/<name>.toml, then config/<name>.local.toml, then environment overrides.

The project root is $SDE_HOME, else the nearest ancestor of the working directory that has
config/settings.toml, else the source checkout this module lives in.
"""

import copy
import json
import logging
import os
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _find_root() -> Path:
    if os.environ.get("SDE_HOME"):
        return Path(os.environ["SDE_HOME"]).expanduser().resolve()
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "config" / "settings.toml").is_file():
            return d
    return Path(__file__).resolve().parents[2]


ROOT = _find_root()


def config_dir() -> Path:
    return Path(os.environ.get("SDE_CONFIG_DIR", ROOT / "config")).expanduser()


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_env_value(raw: str) -> Any:
    """Env overrides are strings; accept JSON so numbers, booleans and lists keep their type."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_env_overrides(cfg: dict, prefixes: dict[str, str], environ: dict[str, str] | None = None,
                        min_parts: int = 2) -> dict:
    """`prefixes` maps an env prefix to a config section, e.g. {"SDE_ROLE__": "roles"}.

    SDE_ROLE__REVIEWER__BACKEND=gateway sets cfg["roles"]["reviewer"]["backend"]. Names are lower-cased.
    """
    environ = os.environ if environ is None else environ
    cfg = copy.deepcopy(cfg)
    for key, raw in environ.items():
        for prefix, section in prefixes.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("__")
            if len(parts) < min_parts or not all(parts):
                continue
            node = cfg.setdefault(section, {})
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = parse_env_value(raw)
    return cfg


_warned: set[str] = set()


def load_layered(name: str, directory: Path | None = None) -> dict:
    """<name>.toml (falling back to the committed <name>.example.toml), then <name>.local.toml."""
    directory = directory or config_dir()
    base_path = directory / f"{name}.toml"
    if not base_path.is_file() and (directory / f"{name}.example.toml").is_file():
        base_path = directory / f"{name}.example.toml"
        if name not in _warned:
            _warned.add(name)
            logging.getLogger("sde_rag.settings").warning(
                "config/%s.toml not found: using %s.example.toml (copy it to %s.toml and adapt it)", name, name, name)
    cfg = tomllib.loads(base_path.read_text()) if base_path.is_file() else {}
    local = Path(os.environ.get(f"SDE_{name.upper()}_CONFIG", directory / f"{name}.local.toml")).expanduser()
    if local.is_file():
        cfg = deep_merge(cfg, tomllib.loads(local.read_text()))
    return cfg


_env_loaded = False


def load_env() -> None:
    """Load the project's .env, then any extra files listed in settings `env_files`. Never overrides."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    load_dotenv(ROOT / ".env", override=False)
    for f in load_layered("settings").get("env_files", []):
        p = Path(f).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            load_dotenv(p, override=False)


@cache
def settings() -> dict:
    load_env()
    # SDE_SETTING__QUERY__CONTEXT_K=8 overrides [query] context_k.
    return apply_env_overrides({"s": load_layered("settings")}, {"SDE_SETTING__": "s"}, min_parts=1)["s"]


def section(name: str) -> dict:
    return settings().get(name, {})


def data_dir() -> Path:
    p = Path(os.environ.get("SDE_DATA_DIR", settings().get("data_dir", "data"))).expanduser()
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def reset_caches() -> None:
    """For tests that swap configuration directories."""
    global _env_loaded
    settings.cache_clear()
    _env_loaded = False
