from dataclasses import dataclass
from typing import List
from enum import Enum



@dataclass
class MiniBatch:
    input_prompts: List
    expected_outputs: List[str]
    inputs: List[str]

class ModelType(Enum):
    QWEN = "qwen"
    GEMMA = "gemma"
    TRANSLATE_GEMMA = "translate_gemma"