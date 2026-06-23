from __future__ import annotations

from dataclasses import asdict

from src.config import get_settings
from src.services.stt_service import STTService
from src.services.translation_service import TranslationService


class MultilingualTranslatorPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.translation_service: TranslationService | None = None
        self.stt_service: STTService | None = None

    def _get_translation_service(self) -> TranslationService:
        if self.translation_service is None:
            self.translation_service = TranslationService(
                model_name=self.settings.translation_model_name,
                device=self.settings.default_device,
            )
        return self.translation_service

    def _get_stt_service(self) -> STTService:
        if self.stt_service is None:
            self.stt_service = STTService(
                model_path=self.settings.stt_model_path,
                device=self.settings.default_device,
            )
        return self.stt_service

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> dict:
        result = self._get_translation_service().translate(text, source_lang, target_lang)
        return asdict(result)

    def translate_audio(
        self,
        audio_path: str,
        audio_language: str,
        target_lang: str,
    ) -> dict:
        stt_result = self._get_stt_service().transcribe(audio_path, source_language=audio_language)
        translated = self._get_translation_service().translate(
            stt_result.text,
            source_lang=audio_language,
            target_lang=target_lang,
        )
        payload = asdict(translated)
        payload["transcribed_text"] = stt_result.text
        payload["audio_path"] = stt_result.audio_path
        return payload
