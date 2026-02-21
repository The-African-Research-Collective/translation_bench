from dataclasses import dataclass
from typing import List


@dataclass
class MiniBatch:
    input_prompts: List
    expected_outputs: List[str]
    inputs: List[str]
