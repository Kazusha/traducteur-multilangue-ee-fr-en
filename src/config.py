from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    stt_model_path: Path | str
    translation_model_name: str
    output_dir: Path
    default_device: str


def _looks_like_local_path(value: str) -> bool:
    """
    Heuristique simple pour distinguer un chemin local d'un identifiant
    Hugging Face Hub (ex: "josiasetnm/crunch_modele_ewe").

    Un identifiant Hub a la forme "namespace/repo_name" sans "./" ni "\\"
    ni lettre de lecteur Windows, et ne correspond Ã  aucun dossier existant
    sur le disque actuel.
    """
    if value.startswith(".") or value.startswith("/") or "\\" in value:
        return True
    if os.path.isabs(value):
        return True
    return False


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]

    stt_model_path_raw = os.getenv("STT_MODEL_PATH", "./crunch_modele_ewe")

    if _looks_like_local_path(stt_model_path_raw):
        stt_model_path: Path | str = Path(stt_model_path_raw)
        if not stt_model_path.is_absolute():
            stt_model_path = (project_root / stt_model_path).resolve()
    else:
        # Identifiant Hugging Face Hub, ex: "josiasetnm/crunch_modele_ewe"
        # -> transmis tel quel Ã  `from_pretrained`, qui le tÃ©lÃ©charge
        # automatiquement depuis le Hub.
        stt_model_path = stt_model_path_raw

    output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=project_root,
        stt_model_path=stt_model_path,
        translation_model_name=os.getenv(
            "TRANSLATION_MODEL_NAME", "facebook/nllb-200-distilled-600M"
        ),
        output_dir=output_dir,
        default_device=os.getenv("DEVICE", "cpu"),
    )
