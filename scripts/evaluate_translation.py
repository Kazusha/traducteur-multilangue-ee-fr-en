from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sacrebleu
from tqdm import tqdm

from src.services.translation_service import TranslationService


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluer un modèle de traduction avec BLEU")
    parser.add_argument("--csv", required=True, help="CSV avec colonnes source,target")
    parser.add_argument("--source-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--source-lang", choices=["fr", "en", "ewe"], required=True)
    parser.add_argument("--target-lang", choices=["fr", "en", "ewe"], required=True)
    parser.add_argument(
        "--model-name",
        default="facebook/nllb-200-distilled-600M",
        help="Nom du modèle HF",
    )
    args = parser.parse_args()

    df = pd.read_csv(Path(args.csv))
    service = TranslationService(model_name=args.model_name)

    hypotheses = []
    references = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_text = str(row[args.source_col])
        ref_text = str(row[args.target_col])
        pred = service.translate(src_text, args.source_lang, args.target_lang).translated_text
        hypotheses.append(pred)
        references.append(ref_text)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU: {bleu.score:.2f}")


if __name__ == "__main__":
    main()
