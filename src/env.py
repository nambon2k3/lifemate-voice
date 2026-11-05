"""Load repo .env values and map them onto agent settings.

Process environment wins over the file. The file wins over configs/agent.yaml.
Docker Compose reads the same file for ports, model names, and model paths.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Environment name -> key stored in the settings dict.
ENV_KEYS = {
    "ASR_URL": "asr_url",
    "ASR_LANGUAGE": "asr_language",
    "ASR_MODEL": "asr_model",
    "PLANNER_MODEL": "planner_model",
    "PLANNER_BASE_URL": "planner_base_url",
    "PLANNER_API_KEY": "planner_api_key",
    "TEMPERATURE": "temperature",
    "MAX_STEPS": "max_steps",
    "SPLIT": "split",
    "KIT_ROOT": "kit_root",
    "CACHE_DIR": "cache_dir",
    "OUTPUT_PATH": "output_path",
    "USE_GOLD_TRANSCRIPT": "use_gold_transcript",
}


def load_dotenv(path: Path | None = None) -> None:
    """Copy KEY=VALUE lines from .env into the process environment.

    Input:
        path (Path | None): dotenv file. Defaults to ``<repo>/.env``.
        A missing file is ignored. A key already set in the process is kept.

    Output:
        None. Fills ``os.environ`` for keys that were unset.
    """
    path = path or (REPO_ROOT / ".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def env_value(name: str, default: str) -> str:
    """Read one configuration variable.

    Input:
        name (str): environment name, such as ``ASR_URL``.
        default (str): value used when the variable is missing or blank.

    Output:
        str: the environment value, or ``default``.
    """
    load_dotenv()
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def apply_env(settings: dict) -> dict:
    """Override settings with the mapped environment variables.

    Input:
        settings (dict): values already read from configs/agent.yaml.

    Output:
        dict: the same dict. A mapped variable that is set, including a blank
        ``ASR_LANGUAGE``, replaces the yaml value. Unset variables leave yaml alone.
    """
    load_dotenv()
    for env_key, setting_key in ENV_KEYS.items():
        if env_key not in os.environ:
            continue
        settings[setting_key] = os.environ[env_key].strip()
    return settings
