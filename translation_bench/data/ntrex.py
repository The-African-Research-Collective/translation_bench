import pandas as pd
from transformers import AutoTokenizer

from torch.utils.data import Dataset
from datasets import Dataset as HfDataset
from datasets import load_dataset
from typing import Dict, List, Optional

from translation_bench.data.data_class import MiniBatch, ModelType
from translation_bench.data.flores_plus import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    REASONING_USER_PROMPT,
)

# NTREX uses the same FLORES+-style config naming convention.
# Each config is a single language variety; examples are aligned by index.
LANGUAGE_NAME_MAPPING = {
    "eng_Latn": "English",
    "eng-US_Latn": "English (US)",
    "eng-GB_Latn": "English (GB)",
    "yor_Latn": "Yoruba",
    "ibo_Latn": "Igbo",
    "hau_Latn": "Hausa",
    "amh_Ethi": "Amharic",
    "swa_Latn": "Swahili",
    "zul_Latn": "Zulu",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "spa_Latn": "Spanish",
    "arb_Arab": "Arabic",
    "zho_Hans": "Chinese (Simplified)",
    "rus_Cyrl": "Russian",
}


def prettify_ntrex_language(config_name: str) -> str:
    if config_name in LANGUAGE_NAME_MAPPING:
        return LANGUAGE_NAME_MAPPING[config_name]
    if "_" in config_name:
        lang, script = config_name.split("_", 1)
        return f"{lang} ({script})"
    return config_name


class NTREXDataset(Dataset):
    """
    Dataset wrapper for davidstap/NTREX.

    NTREX is a multilingual MT evaluation benchmark with 128 language configs.
    Each config contains a single 'text' field; parallel examples are aligned by index.
    Source and target configs are loaded and zipped by position.
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str,
        source_language: str,
        target_language: str,
        num_samples: int,
        tokenizer: AutoTokenizer,
        model_type: ModelType = ModelType.GEMMA,
        hf_token: Optional[str] = None,
        reasoning: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.reasoning = reasoning

        self.data = self._load(
            dataset_name_or_path=dataset_name_or_path,
            split=split,
            source_language=source_language,
            target_language=target_language,
            num_samples=num_samples,
            hf_token=hf_token,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        source_language = item["source_language"]
        target_language = item["target_language"]
        source_text = item["source_text"]
        target_text = item["target_text"]

        if self.model_type == ModelType.TRANSLATE_GEMMA:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"<<<source>>>{source_language}"
                        f"<<<target>>>{target_language}"
                        f"<<<text>>>{source_text}"
                    ),
                }
            ]
        else:
            prompt_template = REASONING_USER_PROMPT if self.reasoning else USER_PROMPT
            user_prompt = prompt_template.format(
                source_language=source_language,
                target_language=target_language,
                source_text=source_text,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

        return {
            "input_prompt": messages,
            "expected_output": target_text,
            "input": source_text,
            "metadata": {
                "id": item["id"],
                "source_config": item["source_config"],
                "target_config": item["target_config"],
            },
        }

    def _load(
        self,
        dataset_name_or_path: str,
        split: str,
        source_language: str,
        target_language: str,
        num_samples: int,
        hf_token: Optional[str] = None,
    ) -> HfDataset:
        src_ds = load_dataset(dataset_name_or_path, source_language, split=split, token=hf_token)
        tgt_ds = load_dataset(dataset_name_or_path, target_language, split=split, token=hf_token)

        src_df = pd.DataFrame(src_ds).rename(columns={"text": "source_text"})
        tgt_df = pd.DataFrame(tgt_ds).rename(columns={"text": "target_text"})

        # NTREX examples are aligned by index (no shared id column)
        merged_df = pd.concat([src_df, tgt_df], axis=1)
        merged_df["id"] = merged_df.index
        merged_df["source_language"] = prettify_ntrex_language(source_language)
        merged_df["target_language"] = prettify_ntrex_language(target_language)
        merged_df["source_config"] = source_language
        merged_df["target_config"] = target_language

        final_df = merged_df[
            ["id", "source_text", "target_text", "source_language", "target_language", "source_config", "target_config"]
        ].reset_index(drop=True)

        dataset = HfDataset.from_pandas(final_df, preserve_index=False)

        if num_samples == -1:
            return dataset
        return dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    @staticmethod
    def collate_fn(batch: List[Dict]) -> MiniBatch:
        return MiniBatch(
            input_prompts=[item["input_prompt"] for item in batch],
            expected_outputs=[item["expected_output"] for item in batch],
            inputs=[item["input"] for item in batch],
        )
