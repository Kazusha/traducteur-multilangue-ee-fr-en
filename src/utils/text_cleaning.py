from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, keep_accents: bool = True) -> str:
    text = normalize_unicode(text)
    text = text.strip()

    if not keep_accents:
        text = unidecode(text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text
