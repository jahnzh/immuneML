from dataclasses import dataclass
from pathlib import Path

@dataclass
class GenerativeModelState:
    exploratory_analysis_units: dict
    result_path: Path = None
    name: str = None