from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import platform

import streamlit as st
# os.environ["HF_HOME"] = "D:/hf_cache"

if platform.system() == "Windows":
    os.environ["HF_HOME"] = "D:/hf_cache"
else:
    os.environ["HF_HOME"] = "/tmp/hf_cache"
    
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.multilingual_pipeline import MultilingualTranslatorPipeline
from src.services.tts_service import is_tts_available, text_to_speech

st.set_page_config(
    page_title="Traducteur FR · EN · EWE",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LANG_OPTIONS = {
    "Français": "fr",
    "Anglais": "en",
    "Ewe": "ewe",
}
LANG_BY_CODE = {v: k for k, v in LANG_OPTIONS.items()}

# ---------------------------------------------------------------------------
# Style — sobre, façon Google Translate (cartes blanches, peu de bruit visuel)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        color: #5f6368;
        font-size: 0.95rem;
        margin-bottom: 1.4rem;
    }
    .lang-card {
        background: #ffffff;
        border: 1px solid #e3e3e3;
        border-radius: 12px;
        padding: 0.6rem 0.9rem 0.9rem 0.9rem;
    }
    .lang-card-label {
        font-size: 0.8rem;
        color: #5f6368;
        font-weight: 500;
        margin-bottom: 0.2rem;
    }
    .result-empty {
        color: #9aa0a6;
        font-style: italic;
    }
    .fallback-warning {
        font-size: 0.82rem;
        color: #b06000;
        background: #fff6e5;
        border-radius: 6px;
        padding: 0.35rem 0.6rem;
        margin-top: 0.4rem;
        display: inline-block;
    }
    div[data-testid="stTextArea"] textarea {
        border: none !important;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Traducteur Français · Anglais · Ewe</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Texte ou voix — avec reconnaissance vocale ewe et lecture audio des traductions.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# État
# ---------------------------------------------------------------------------
st.session_state.setdefault("text_source_label", "Français")
st.session_state.setdefault("text_target_label", "Anglais")
st.session_state.setdefault("text_input", "")
st.session_state.setdefault("text_output", "")

st.session_state.setdefault("voice_source_label", "Ewe")
st.session_state.setdefault("voice_target_label", "Français")
st.session_state.setdefault("voice_transcription", "")
st.session_state.setdefault("voice_translation", "")


def rerun_app() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
    experimental_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun_fn):
        experimental_rerun_fn()


@st.cache_resource(show_spinner=False)
def load_pipeline() -> MultilingualTranslatorPipeline:
    return MultilingualTranslatorPipeline()


def get_pipeline() -> MultilingualTranslatorPipeline | None:
    try:
        return load_pipeline()
    except Exception as exc:
        st.error("Impossible de charger les modèles.")
        st.exception(exc)
        return None


def save_audio_to_temp_file(audio_file) -> str:
    suffix = Path(audio_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getvalue())
        return tmp.name


def render_listen_button(text: str, lang_code: str, key: str) -> None:
    """Affiche un bouton 'Écouter' qui génère et joue l'audio.

    Utilise gTTS pour le français/anglais et le modèle Coqui VITS
    natif ewe (entraîné sur le dataset OpenBible) pour l'ewe.
    """
    if not text or not text.strip():
        return
    if not is_tts_available(lang_code):
        return

    if st.button("🔊 Écouter", key=key):
        spinner_msg = (
            "Chargement du modèle vocal ewe (première fois)..."
            if lang_code == "ewe"
            else "Génération de l'audio..."
        )
        with st.spinner(spinner_msg):
            speech = text_to_speech(text, lang_code)
        if speech.available:
            with open(speech.audio_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(speech.audio_path)
            audio_format = "audio/wav" if speech.audio_path.endswith(".wav") else "audio/mp3"
            st.audio(audio_bytes, format=audio_format)
        else:
            st.info("Synthèse vocale non disponible pour cette langue.")


tab_text, tab_audio = st.tabs(["📝 Texte", "🎙️ Voix"])

# ---------------------------------------------------------------------------
# Onglet Texte
# ---------------------------------------------------------------------------
with tab_text:
    lang_left, lang_swap, lang_right = st.columns([4, 0.6, 4], gap="small")

    with lang_left:
        st.selectbox("Langue source", list(LANG_OPTIONS.keys()), key="text_source_label", label_visibility="collapsed")

    with lang_swap:
        if st.button("⇄", key="swap_text_langs", use_container_width=True):
            st.session_state["text_source_label"], st.session_state["text_target_label"] = (
                st.session_state["text_target_label"],
                st.session_state["text_source_label"],
            )
            st.session_state["text_input"], st.session_state["text_output"] = (
                st.session_state.get("text_output", ""),
                st.session_state.get("text_input", ""),
            )
            rerun_app()

    with lang_right:
        st.selectbox("Langue cible", list(LANG_OPTIONS.keys()), key="text_target_label", label_visibility="collapsed")

    source_col, target_col = st.columns(2, gap="medium")

    with source_col:
        st.markdown('<div class="lang-card">', unsafe_allow_html=True)
        st.text_area(
            "Texte source",
            key="text_input",
            height=200,
            placeholder="Écrivez ici ce que vous voulez traduire...",
            label_visibility="collapsed",
        )
        render_listen_button(
            st.session_state.get("text_input", ""),
            LANG_OPTIONS[st.session_state["text_source_label"]],
            key="listen_source_text",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with target_col:
        st.markdown('<div class="lang-card">', unsafe_allow_html=True)
        output_text = st.session_state.get("text_output", "")
        st.text_area(
            "Texte traduit",
            value=output_text,
            height=200,
            disabled=True,
            label_visibility="collapsed",
            placeholder="La traduction apparaîtra ici.",
        )
        render_listen_button(
            output_text,
            LANG_OPTIONS[st.session_state["text_target_label"]],
            key="listen_target_text",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    action_translate, action_clear, _ = st.columns([1.3, 1.0, 5.0])
    with action_translate:
        translate_text_clicked = st.button("Traduire", type="primary", key="translate_text_button")
    with action_clear:
        clear_text_clicked = st.button("Effacer", key="clear_text_button")

    if clear_text_clicked:
        st.session_state["text_input"] = ""
        st.session_state["text_output"] = ""
        rerun_app()

    if translate_text_clicked:
        source_label = st.session_state["text_source_label"]
        target_label = st.session_state["text_target_label"]
        source_text = st.session_state["text_input"].strip()

        if not source_text:
            st.warning("Saisis un texte à traduire.")
        elif LANG_OPTIONS[source_label] == LANG_OPTIONS[target_label]:
            st.warning("La langue source et cible doivent être différentes.")
        else:
            pipeline = get_pipeline()
            if pipeline is not None:
                with st.spinner("Traduction en cours..."):
                    try:
                        result = pipeline.translate_text(
                            text=source_text,
                            source_lang=LANG_OPTIONS[source_label],
                            target_lang=LANG_OPTIONS[target_label],
                        )
                    except Exception as exc:
                        st.error("Erreur pendant la traduction.")
                        st.exception(exc)
                    else:
                        st.session_state["text_output"] = result["translated_text"]
                        rerun_app()

# ---------------------------------------------------------------------------
# Onglet Voix
# ---------------------------------------------------------------------------
with tab_audio:
    audio_lang_left, audio_lang_right = st.columns(2, gap="large")

    with audio_lang_left:
        st.selectbox("Langue parlée", list(LANG_OPTIONS.keys()), key="voice_source_label")

    with audio_lang_right:
        st.selectbox("Langue cible de traduction", list(LANG_OPTIONS.keys()), key="voice_target_label")

    st.markdown("**Entrée vocale**")
    recorded_audio = None
    if hasattr(st, "audio_input"):
        recorded_audio = st.audio_input("Parle directement avec ton micro")
        if recorded_audio is not None:
            st.audio(recorded_audio)
    else:
        st.info("La capture micro n'est pas disponible dans cette version de Streamlit. Utilise l'import de fichier.")

    uploaded_audio = st.file_uploader(
        "Ou importe un fichier audio (.wav, .mp3, .m4a)",
        type=["wav", "mp3", "m4a"],
        key="voice_audio_upload",
    )

    selected_audio = recorded_audio if recorded_audio is not None else uploaded_audio

    audio_action_translate, audio_action_clear, _ = st.columns([2.2, 1.3, 3.5])
    with audio_action_translate:
        translate_voice_clicked = st.button("Transcrire et traduire", type="primary", key="translate_voice_button")
    with audio_action_clear:
        clear_voice_clicked = st.button("Effacer résultat", key="clear_voice_button")

    if clear_voice_clicked:
        st.session_state["voice_transcription"] = ""
        st.session_state["voice_translation"] = ""
        rerun_app()

    if translate_voice_clicked:
        source_label = st.session_state["voice_source_label"]
        target_label = st.session_state["voice_target_label"]

        if selected_audio is None:
            st.warning("Ajoute un enregistrement micro ou un fichier audio.")
        elif LANG_OPTIONS[source_label] == LANG_OPTIONS[target_label]:
            st.warning("La langue source et cible doivent être différentes.")
        else:
            pipeline = get_pipeline()
            if pipeline is not None:
                temp_audio_path = None
                try:
                    temp_audio_path = save_audio_to_temp_file(selected_audio)
                    with st.spinner("Transcription et traduction audio en cours..."):
                        result = pipeline.translate_audio(
                            audio_path=temp_audio_path,
                            audio_language=LANG_OPTIONS[source_label],
                            target_lang=LANG_OPTIONS[target_label],
                        )
                except Exception as exc:
                    st.error("Erreur pendant le traitement audio.")
                    st.exception(exc)
                else:
                    st.session_state["voice_transcription"] = result["transcribed_text"]
                    st.session_state["voice_translation"] = result["translated_text"]
                    rerun_app()
                finally:
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)

    transcription_col, translation_col = st.columns(2, gap="large")
    with transcription_col:
        st.markdown('<div class="lang-card">', unsafe_allow_html=True)
        st.markdown("**Transcription**")
        transcription_text = st.session_state.get("voice_transcription", "")
        st.text_area(
            "Transcription audio",
            value=transcription_text,
            height=160,
            disabled=True,
            label_visibility="collapsed",
        )
        render_listen_button(
            transcription_text,
            LANG_OPTIONS[st.session_state["voice_source_label"]],
            key="listen_voice_transcription",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with translation_col:
        st.markdown('<div class="lang-card">', unsafe_allow_html=True)
        st.markdown("**Traduction**")
        translation_text = st.session_state.get("voice_translation", "")
        st.text_area(
            "Traduction audio",
            value=translation_text,
            height=160,
            disabled=True,
            label_visibility="collapsed",
        )
        render_listen_button(
            translation_text,
            LANG_OPTIONS[st.session_state["voice_target_label"]],
            key="listen_voice_translation",
        )
        st.markdown("</div>", unsafe_allow_html=True)