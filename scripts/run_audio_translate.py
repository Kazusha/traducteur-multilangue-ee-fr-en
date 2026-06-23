from __future__ import annotations

import argparse

from src.pipelines.multilingual_pipeline import MultilingualTranslatorPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio -> STT -> Traduction FR/EN/EWE")
    parser.add_argument("--audio", required=True, help="Chemin audio")
    parser.add_argument("--source", choices=["fr", "en", "ewe"], required=True)
    parser.add_argument("--target", choices=["fr", "en", "ewe"], required=True)
    args = parser.parse_args()

    pipeline = MultilingualTranslatorPipeline()
    result = pipeline.translate_audio(
        audio_path=args.audio,
        audio_language=args.source,
        target_lang=args.target,
    )
    print("Transcription:", result["transcribed_text"])
    print("Traduction:", result["translated_text"])


if __name__ == "__main__":
    main()
