from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.text_cleaning import clean_text


LANG_COLUMNS = {
    "fr": "fr_text",
    "en": "en_text",
    "ewe": "ewe_text",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoyage corpus parallèle FR/EN/EWE")
    parser.add_argument("--input", required=True, help="CSV brut")
    parser.add_argument("--output", required=True, help="CSV nettoyé")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    expected = [LANG_COLUMNS["fr"], LANG_COLUMNS["en"], LANG_COLUMNS["ewe"]]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    for col in expected:
        df[col] = df[col].fillna("").map(clean_text)

    df = df[(df[expected[0]] != "") & (df[expected[1]] != "") & (df[expected[2]] != "")]
    df = df.drop_duplicates(subset=expected).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Fichier nettoyé: {output_path} ({len(df)} lignes)")


if __name__ == "__main__":
    main()
