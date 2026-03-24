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

# AfriMTE language pair configs available in masakhane/AfriMTE-WMT2024.
# Format: '<src_code>-<tgt_code>' e.g. 'eng-yor', 'yor-eng'
# The dataset provides source, hypothesis (pre-generated MT output), reference,
# and a human quality score. For translation benchmarking we use source → reference.
AFRIMTE_CONFIGS = [
    "ary-fra", "eng-arz", "eng-fra", "eng-hau", "eng-ibo",
    "eng-kik", "eng-luo", "eng-som", "eng-swh", "eng-twi",
    "eng-xho", "eng-yor", "yor-eng",
]


class AfriMTEDataset(Dataset):
    """
    Dataset wrapper for masakhane/AfriMTE-WMT2024.

    AfriMTE is a human-evaluated MT quality dataset for African languages.
    Each example contains:
      - source: original source sentence
      - hypothesis: a pre-existing MT system output (not used for generation)
      - reference: human reference translation
      - score: human quality score (MQM-style)
      - source_language / target_language: language names

    For translation benchmarking, the model translates `source` and we
    evaluate against `reference`. The `score` and `hypothesis` fields are
    preserved in metadata for downstream analysis.

    Available configs: ary-fra, eng-arz, eng-fra, eng-hau, eng-ibo,
                       eng-kik, eng-luo, eng-som, eng-swh, eng-twi,
                       eng-xho, eng-yor, yor-eng
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str,
        subset: str,
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
            subset=subset,
            num_samples=num_samples,
            hf_token=hf_token,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        source_language = item["source_language"]
        target_language = item["target_language"]
        source_text = item["source"]
        target_text = item["reference"]

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
                "language_pair": item["language_pair"],
                "human_score": item["score"],
                "existing_hypothesis": item["hypothesis"],
            },
        }

    def _load(
        self,
        dataset_name_or_path: str,
        split: str,
        subset: str,
        num_samples: int,
        hf_token: Optional[str] = None,
    ) -> HfDataset:
        ds = load_dataset(dataset_name_or_path, subset, split=split, token=hf_token)

        if num_samples == -1:
            return ds
        return ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    @staticmethod
    def collate_fn(batch: List[Dict]) -> MiniBatch:
        return MiniBatch(
            input_prompts=[item["input_prompt"] for item in batch],
            expected_outputs=[item["expected_output"] for item in batch],
            inputs=[item["input"] for item in batch],
        )
