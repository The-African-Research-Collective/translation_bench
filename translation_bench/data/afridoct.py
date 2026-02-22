import pandas as pd
from enum import Enum
from transformers import AutoTokenizer

from torch.utils.data import Dataset
from datasets import Dataset as HfDataset
from datasets import load_dataset
from jinja2 import Environment
from typing import Dict, List

from translation_bench.data.data_class import MiniBatch

SYSTEM_PROMPT = """You are a helpful assistant for translating documents for low-resource languages"""

USER_PROMPT = """You are given a document in {source_language}. Your task is to translate the document into {target_language}.gs

Here is the document to be translated:
{source_text}

Translation
"""

MAPPING = {
    "sw": "Swahili",
    "am": "Amharic",
    "yo": "Yoruba",
    "ha": "Hausa",
    "zu": "Zulu",
    "en": "English",
}

class ModelType(Enum):
    QWEN = "qwen"
    GEMMA = "gemma"
    TRANSLATE_GEMMA = "translate_gemma"


class AfriDocMTDataset(Dataset):
    def __init__(
        self,
        dataset_name_or_path: str,
        split: str,
        subset: str,
        num_samples: str,
        tokenizer: AutoTokenizer,
        source_language: str,
        target_language: str,
        model_type: ModelType = ModelType.GEMMA,
    ):
        self.tokenizer = tokenizer
        self.data = self.load_dataset_hgf(
            dataset_name_or_path,
            split,
            subset,
            num_samples,
            source_language,
            target_language,
        )

        self.chat_template = Environment().from_string(self.tokenizer.chat_template)
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        source_language = item["source_language"]
        target_language = item["target_language"]
        source_text = item["source_text"]
        target_text = item["target_text"]

        if self.model_type == ModelType.TRANSLATE_GEMMA:
            messages = [
                {
                    "role": "user",
                    "content": f"<<<source>>>{source_language}<<<target>>>{target_language}<<<text>>>{source_text}"
                }
            ]

            return {
                "input_prompt": messages,
                "expected_output": target_text,
                "input": source_text,
            }
        
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
            }

    def load_dataset_hgf(
        self,
        dataset_name_or_path: str,
        split: str,
        subset: str,
        num_samples: str,
        source_language: list,
        target_language: str,
    ):
        all_data = load_dataset(dataset_name_or_path, subset)[split]
        df = pd.DataFrame(all_data)

        total = []

        trimmed_df = df[[source_language, target_language]]
        trimmed_df = trimmed_df.rename(columns={source_language: "source_text", target_language: "target_text"})
        trimmed_df["source_language"] = MAPPING[source_language]
        trimmed_df["target_language"] = MAPPING[target_language]

        total.append(trimmed_df)

        merged_df = pd.concat(total, ignore_index=True)
        selected_dataset = HfDataset.from_pandas(merged_df)

        if num_samples == -1:
            return selected_dataset
        else:
            return selected_dataset.shuffle(seed=42).select(range(num_samples))

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
    dataset = AfriDocMTDataset(
        dataset_name_or_path="masakhane/AfriDocMT",
        split="train",
        subset="doc_health_5",
        num_samples=100,
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct"),
        source_language="yo",
        target_language="en",
    )

    print(dataset[0])

    # dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    # for batch in dataloader:
    #     print(batch)
    #     break