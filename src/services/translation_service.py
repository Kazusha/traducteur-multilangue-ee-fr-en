from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG_CODE_MAP = {
    "fr": "fra_Latn",
    "en": "eng_Latn",
    "ewe": "ewe_Latn",
}


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str


class TranslationService:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 256,
    ) -> TranslationResult:
        source_code = LANG_CODE_MAP[source_lang]
        target_code = LANG_CODE_MAP[target_lang]

        self.tokenizer.src_lang = source_code
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code),
                max_new_tokens=max_new_tokens,
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
        )
