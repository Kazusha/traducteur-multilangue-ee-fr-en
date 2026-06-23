from __future__ import annotations

import argparse

from src.pipelines.multilingual_pipeline import MultilingualTranslatorPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Traduction texte FR/EN/EWE")
    parser.add_argument("--text", required=True, help="Texte à traduire")
    parser.add_argument("--source", choices=["fr", "en", "ewe"], required=True)
    parser.add_argument("--target", choices=["fr", "en", "ewe"], required=True)
    args = parser.parse_args()

    pipeline = MultilingualTranslatorPipeline()
    result = pipeline.translate_text(
        text=args.text,
        source_lang=args.source,
        target_lang=args.target,
    )
    print(result["translated_text"])


if __name__ == "__main__":
    main()
