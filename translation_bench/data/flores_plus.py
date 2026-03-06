import pandas as pd
from transformers import AutoTokenizer

from torch.utils.data import Dataset
from datasets import Dataset as HfDataset
from datasets import load_dataset
from jinja2 import Environment
from typing import Dict, List, Optional

from translation_bench.data.data_class import MiniBatch, ModelType


SYSTEM_PROMPT = """You are a helpful assistant for translating documents for low-resource languages."""

USER_PROMPT = """You are given a document in {source_language}. Your task is to translate the document into {target_language}.

Here is the document to be translated:
{source_text}

Translation
"""


# Optional pretty names for a few common FLORES+ configs.
# Keys here are FLORES+ subset/config names.
LANGUAGE_NAME_MAPPING = {
    "eng_Latn": "English",
    "yor_Latn": "Yoruba",
    "ibo_Latn": "Igbo",
    "hau_Latn": "Hausa",
    "amh_Ethi": "Amharic",
    "swh_Latn": "Swahili",
    "zul_Latn": "Zulu",
    "fra_Latn": "French",
}


def prettify_flores_language(config_name: str) -> str:
    """
    Convert FLORES+ config names like 'yor_Latn' to a readable label.
    Falls back gracefully if the config is not in LANGUAGE_NAME_MAPPING.
    """
    if config_name in LANGUAGE_NAME_MAPPING:
        return LANGUAGE_NAME_MAPPING[config_name]

    if "_" in config_name:
        lang, script = config_name.split("_", 1)
        return f"{lang} ({script})"
    return config_name


class FloresPlusDataset(Dataset):
    """
    Dataset wrapper for openlanguagedata/flores_plus.

    Important:
    - FLORES+ is intended primarily as an evaluation benchmark, not training data.
    - Each config/subset corresponds to a single language variety, e.g. 'eng_Latn', 'yor_Latn'.
    - Parallel examples are aligned by ('id', 'split') across language configs.
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
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.chat_template = Environment().from_string(self.tokenizer.chat_template)

        self.data = self.load_dataset_hgf(
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
            user_prompt = USER_PROMPT.format(
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
                "split": item["split"],
                "source_config": item["source_config"],
                "target_config": item["target_config"],
                "domain": item.get("domain"),
                "topic": item.get("topic"),
                "url": item.get("url"),
            },
        }

    def load_dataset_hgf(
        self,
        dataset_name_or_path: str,
        split: str,
        source_language: str,
        target_language: str,
        num_samples: int,
        hf_token: Optional[str] = None,
    ) -> HfDataset:
        """
        Load source and target FLORES+ language configs and join them on `id`.

        Example:
            source_language='yor_Latn'
            target_language='eng_Latn'
        """

        # Each config is one language variety in FLORES+.
        src_ds = load_dataset(
            dataset_name_or_path,
            source_language,
            split=split,
            token=hf_token,
        )
        tgt_ds = load_dataset(
            dataset_name_or_path,
            target_language,
            split=split,
            token=hf_token,
        )

        src_df = pd.DataFrame(src_ds)
        tgt_df = pd.DataFrame(tgt_ds)

        # Keep only fields we need. Joining on `id` is enough because split is already fixed,
        # but we keep split for metadata clarity.
        src_df = src_df.rename(
            columns={
                "text": "source_text",
                "url": "source_url",
                "domain": "source_domain",
                "topic": "source_topic",
            }
        )
        tgt_df = tgt_df.rename(
            columns={
                "text": "target_text",
                "url": "target_url",
                "domain": "target_domain",
                "topic": "target_topic",
            }
        )

        merged_df = src_df.merge(
            tgt_df[["id", "target_text"]],
            on="id",
            how="inner",
        )

        # Use source-side metadata fields. In FLORES+ these should line up across languages.
        merged_df["source_language"] = prettify_flores_language(source_language)
        merged_df["target_language"] = prettify_flores_language(target_language)
        merged_df["source_config"] = source_language
        merged_df["target_config"] = target_language

        # Normalize a few metadata fields for convenience
        merged_df["url"] = merged_df.get("source_url")
        merged_df["domain"] = merged_df.get("source_domain")
        merged_df["topic"] = merged_df.get("source_topic")

        final_df = merged_df[
            [
                "id",
                "split",
                "source_text",
                "target_text",
                "source_language",
                "target_language",
                "source_config",
                "target_config",
                "url",
                "domain",
                "topic",
            ]
        ].reset_index(drop=True)

        selected_dataset = HfDataset.from_pandas(final_df, preserve_index=False)

        if num_samples == -1:
            return selected_dataset
        return selected_dataset.shuffle(seed=42).select(
            range(min(num_samples, len(selected_dataset)))
        )

    @staticmethod
    def collate_fn(batch: List[Dict]) -> MiniBatch:
        input_prompts = [item["input_prompt"] for item in batch]
        expected_outputs = [item["expected_output"] for item in batch]
        inputs = [item["input"] for item in batch]

        return MiniBatch(
            input_prompts=input_prompts,
            expected_outputs=expected_outputs,
            inputs=inputs,
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    dataset = FloresPlusDataset(
        dataset_name_or_path="openlanguagedata/flores_plus",
        split="dev",
        source_language="yor_Latn",
        target_language="eng_Latn",
        num_samples=10,
        tokenizer=tokenizer,
        model_type=ModelType.GEMMA,
        hf_token=None,  # or set your HF token string here
    )

    print(dataset[0])
