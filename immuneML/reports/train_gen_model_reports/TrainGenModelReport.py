from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.ml_methods.generative_models.GenerativeModel import GenerativeModel
from immuneML.reports.Report import Report
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder


class TrainGenModelReport(Report):
    """
    TrainGenModel reports show some type of features or statistics comparing two datasets: the original and generated
    one, potentially in combination with the trained model. These reports can only be used inside TrainGenModel
    instruction with the aim of comparing two datasets: the dataset used to train a generative model and the dataset
    created from the trained model.
    """

    def __init__(self, original_dataset: Dataset = None, generated_dataset: Dataset = None, result_path: Path = None,
                 name: str = None, number_of_processes: int = 1, model: GenerativeModel = None):
        """
        The arguments defined below are set at runtime by the instruction.
        Concrete classes inheriting DataComparisonReport may include additional parameters that will be set by the user
        in the form of input arguments (e.g., from the YAML file).

        Args:

            original_dataset (Dataset): a dataset object (can be repertoire, receptor or sequence dataset, depending
            on the specific report) provided as input to the TrainGenModel instruction

            generated_dataset (Dataset): a dataset object as produced from the generative model after being trained on
            the original dataset

            result_path (Path): location where the results (plots, tables, etc.) will be stored

            name (str): user-defined name of the report used in the HTML overview automatically generated by the
            platform from the key used to define the report in the YAML

            number_of_processes (int): how many processes should be created at once to speed up the analysis.
            For personal machines, 4 or 8 is usually a good choice.

            model (GenerativeModel): trained generative model from the instruction

        """
        super().__init__(name=name, number_of_processes=number_of_processes)
        self.original_dataset = original_dataset
        self.generated_dataset = generated_dataset
        self.model = model
        self.result_path = result_path

    @staticmethod
    def get_title():
        return "TrainGenModel reports"
