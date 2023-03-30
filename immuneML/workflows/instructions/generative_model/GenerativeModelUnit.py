from dataclasses import dataclass
from pathlib import Path
import numpy

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.ml_reports.GeneratorReport import GeneratorReport
from immuneML.reports.ReportResult import ReportResult
from immuneML.ml_methods.GenerativeModel import GenerativeModel


@dataclass
class GenerativeModelUnit:

    report: GeneratorReport
    genModel: GenerativeModel
    amount: int = 10
    dataset: Dataset = None
    generated_sequences: list = None
    encoder: DatasetEncoder = None
    label_config: LabelConfiguration = None
    report_result: ReportResult = None
    path: Path = None
    predictions_path: Path = None