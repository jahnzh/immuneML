from pathlib import Path

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.environment.Label import Label
from immuneML.hyperparameter_optimization.HPSetting import HPSetting
from immuneML.ml_methods.MLMethod import MLMethod
from immuneML.reports.Report import Report


class MLReport(Report):
    '''
    ML model reports show some type of features or statistics about one trained ML model.

    In the :ref:`TrainMLModel` instruction, ML model reports can be specified inside the 'selection' or 'assessment' specification under the key 'reports/models'.
    Example:

    .. indent with spaces
    .. code-block:: yaml

        my_instruction:
            type: TrainMLModel
            selection:
                reports:
                    models:
                        - my_ml_report
                # other parameters...
            assessment:
                reports:
                    models:
                        - my_ml_report
                # other parameters...
            # other parameters...
    '''

    def __init__(self, train_dataset: Dataset = None, test_dataset: Dataset = None, method: MLMethod = None,
                 result_path: Path = None, name: str = None, hp_setting: HPSetting = None, label: Label =None, number_of_processes: int = 1):
        '''
        The arguments defined below are set at runtime by the instruction.
        Concrete classes inheriting MLReport may include additional parameters that will be set by the user in the form of input arguments.

        train_dataset (Dataset): a dataset object (repertoire, receptor or sequence dataset) with encoded_data attribute set to an EncodedData object that was used for training the ML method
        test_dataset (Dataset): same as train_dataset, except it is not used for training and then maybe be used for testing the method
        method (MLMethod): a trained instance of a concrete subclass of MLMethod object
        result_path (Path): location where the report results will be stored
        hp_setting (HPSetting): a HPSetting object describing the ML method, encoding and preprocessing used
        label (Label): the label for which the model was trained
        name (str): user-defined name of the report used in the HTML overview automatically generated by the platform
        number_of_processes (int): how many processes should be created at once to speed up the analysis. For personal machines, 4 or 8 is usually a good choice.
        '''
        super().__init__(name, number_of_processes)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.method = method
        self.result_path = result_path
        self.name = name
        self.label = label

    @staticmethod
    def get_title():
        return "ML model reports"
