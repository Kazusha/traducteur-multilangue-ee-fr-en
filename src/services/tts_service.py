from __future__ import annotations
import streamlit as st
import tempfile
from dataclasses import dataclass

from gtts import gTTS

# Langues supportées nativement par gTTS.
GTTS_LANGS = {
    "fr": "fr",
    "en": "en",
}

# Modèle Coqui VITS entraîné nativement en ewe (dataset OpenBible).
EWE_TTS_MODEL_NAME = "tts_models/ewe/openbible/vits"

@st.cache_resource
def load_ewe_model():
    from TTS.api import TTS
    return TTS(EWE_TTS_MODEL_NAME)

@dataclass
class SpeechResult:
    audio_path: str
    lang: str
    available: bool
    is_phonetic_fallback: bool = False

def text_to_speech(text: str, lang: str) -> SpeechResult:
    if not text or not text.strip():
        return SpeechResult("", lang, False)

    try:
        if lang in GTTS_LANGS:
            tts = gTTS(text=text, lang=GTTS_LANGS[lang])

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            return SpeechResult(tmp.name, lang, True)

        if lang == "ewe":
            model = load_ewe_model()

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            model.tts_to_file(text=text, file_path=tmp.name)

            return SpeechResult(tmp.name, lang, True)

    except Exception as e:
        print("TTS error:", e)

    return SpeechResult("", lang, False)
# Instance Coqui chargée à la demande (lazy) et mise en cache en mémoire,
# pour ne pas recharger le modèle à chaque appel.
_ewe_tts_model = None


# def _get_ewe_tts_model():
#     global _ewe_tts_model
#     if _ewe_tts_model is None:
#         from TTS.api import TTS  # import différé : coqui-tts est lourd à charger

#         _ewe_tts_model = TTS(EWE_TTS_MODEL_NAME)
#     return _ewe_tts_model
def _get_ewe_tts_model():
    global _ewe_tts_model
    if _ewe_tts_model is None:
        # Patch compatibilité transformers>=4.45.0 — isin_mps_friendly supprimé
        try:
            import transformers.pytorch_utils as pu
            if not hasattr(pu, "isin_mps_friendly"):
                import torch
                pu.isin_mps_friendly = torch.isin
        except Exception:
            pass
        from TTS.api import TTS
        _ewe_tts_model = TTS(EWE_TTS_MODEL_NAME)
    return _ewe_tts_model

@dataclass
class SpeechResult:
    audio_path: str
    lang: str
    available: bool
    is_phonetic_fallback: bool = False


def is_tts_available(lang: str) -> bool:
    """Indique si une voix peut être générée pour cette langue."""
    return lang in GTTS_LANGS or lang == "ewe"


def text_to_speech(text: str, lang: str) -> SpeechResult:
    """
    Génère un fichier audio à partir d'un texte.

    - fr / en -> voix native gTTS (sortie .mp3).
    - ewe     -> voix native Coqui VITS entraînée sur l'ewe (sortie .wav).
    - autre   -> available=False.

    L'appelant est responsable de supprimer le fichier temporaire généré
    une fois qu'il n'en a plus besoin.
    """
    if not text or not text.strip():
        return SpeechResult(audio_path="", lang=lang, available=False)

    if lang in GTTS_LANGS:
        gtts_lang = GTTS_LANGS[lang]
        tts = gTTS(text=text, lang=gtts_lang)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        tmp.close()

        return SpeechResult(audio_path=tmp.name, lang=lang, available=True)

    if lang == "ewe":
        model = _get_ewe_tts_model()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        model.tts_to_file(text=text, file_path=tmp.name)

        return SpeechResult(audio_path=tmp.name, lang=lang, available=True)

    return SpeechResult(audio_path="", lang=lang, available=False)