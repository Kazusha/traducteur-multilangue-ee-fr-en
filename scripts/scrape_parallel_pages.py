from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


def extract_paragraphs(url: str) -> list[str]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return [p for p in paragraphs if p]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scraping simple de pages parallèles")
    parser.add_argument("--fr-url", required=True)
    parser.add_argument("--en-url", required=True)
    parser.add_argument("--ewe-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fr = extract_paragraphs(args.fr_url)
    en = extract_paragraphs(args.en_url)
    ewe = extract_paragraphs(args.ewe_url)

    n = min(len(fr), len(en), len(ewe))
    if n == 0:
        raise ValueError("Aucune donnée extraite, vérifier les URLs")

    df = pd.DataFrame(
        {
            "fr_text": fr[:n],
            "en_text": en[:n],
            "ewe_text": ewe[:n],
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Corpus sauvegardé: {out} ({len(df)} lignes)")


if __name__ == "__main__":
    main()
