# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path

# import librosa
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# LANG_PROMPT_MAP = {
#     "fr": "french",
#     "en": "english",
#     # Ewe n'est pas une langue native Whisper: on laisse le modèle décider
#     # sans contrainte forcée de langue.
#     "ewe": None,
# }


# @dataclass
# class TranscriptionResult:
#     audio_path: str
#     text: str


# class STTService:
#     def __init__(self, model_path: Path, device: str = "cpu") -> None:
#         self.device = device
#         self.processor = AutoProcessor.from_pretrained(model_path)
#         self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
#         self.model.to(device)

#     def transcribe(self, audio_path: str, source_language: str = "en") -> TranscriptionResult:
#         speech, sample_rate = librosa.load(audio_path, sr=16000)
#         inputs = self.processor(
#             speech,
#             sampling_rate=sample_rate,
#             return_tensors="pt",
#         )

#         input_features = inputs.input_features.to(self.device)

#         whisper_language = LANG_PROMPT_MAP.get(source_language)
#         if whisper_language is None:
#             forced_decoder_ids = None
#         else:
#             forced_decoder_ids = self.processor.get_decoder_prompt_ids(
#                 language=whisper_language,
#                 task="transcribe",
#             )

#         with torch.no_grad():
#             predicted_ids = self.model.generate(
#                 input_features,
#                 forced_decoder_ids=forced_decoder_ids,
#                 max_new_tokens=256,
#             )

#         transcription = self.processor.batch_decode(
#             predicted_ids,
#             skip_special_tokens=True,
#         )[0]

#         return TranscriptionResult(audio_path=audio_path, text=transcription.strip())
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
)

LANG_PROMPT_MAP = {
    "fr": "french",
    "en": "english",
    "ewe": None,
}


@dataclass
class TranscriptionResult:
    audio_path: str
    text: str


class STTService:
    def __init__(self, model_path: Path | str, device: str = "cpu") -> None:
        self.device = device

        # Chargement manuel pour eviter le doublon feature_extractor
        # que AutoProcessor.from_pretrained() cause avec les modeles
        # dont le processor_config.json contient deja feature_extractor.
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        tokenizer = WhisperTokenizer.from_pretrained(model_path)
        self.processor = WhisperProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
        self.model.to(device)

    def transcribe(self, audio_path: str, source_language: str = "en") -> TranscriptionResult:
        speech, sample_rate = librosa.load(audio_path, sr=16000)
        inputs = self.processor(
            speech,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device)

        whisper_language = LANG_PROMPT_MAP.get(source_language)
        if whisper_language is None:
            forced_decoder_ids = None
        else:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=whisper_language,
                task="transcribe",
            )

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256,
            )

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return TranscriptionResult(audio_path=audio_path, text=transcription.strip())