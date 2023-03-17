import abc
import os
import warnings
from pathlib import Path

import dill
import pkg_resources
import yaml
import numpy as np
from sklearn.utils.validation import check_is_fitted

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.environment.Label import Label
from immuneML.ml_methods.UnsupervisedMLMethod import UnsupervisedMLMethod
from immuneML.util.FilenameHandler import FilenameHandler
from immuneML.util.PathBuilder import PathBuilder


class GenerativeModel(UnsupervisedMLMethod):

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        super(GenerativeModel, self).__init__()
        self.model = None

        if parameter_grid is not None and "show_warnings" in parameter_grid:
            self.show_warnings = parameter_grid.pop("show_warnings")[0]
        elif parameters is not None and "show_warnings" in parameters:
            self.show_warnings = parameters.pop("show_warnings")
        else:
            self.show_warnings = True

        self._parameter_grid = parameter_grid
        self._parameters = parameters
        self._length_of_sequence = 20 #Set as average, will be overwritten
        self.feature_names = None
        self.class_mapping = None
        self.label = None
        self.alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.char2idx = {u: i for i, u in enumerate(self.alphabet)}
        self.idx2char = np.array(self.alphabet)

    def fit(self, encoded_data, cores_for_training: int = 2, result_path: Path = None):
        self._length_of_sequence = 21# if "length_of_sequence" not in encoded_data.info else encoded_data.info["length_of_sequence"]
        self.model = self._fit(encoded_data.examples, cores_for_training, result_path)

    def generate(self, amount=10, path_to_model: Path = None):
        pass

    @abc.abstractmethod
    def _fit(self, X, cores_for_training: int = 1):
        pass


    def check_is_fitted(self, label_name: str):
        if self.label.name == label_name or label_name is None:
            return check_is_fitted(self.model, ["estimators_", "coef_", "estimator", "_fit_X", "dual_coef_"], all_or_any=any)


    def store(self, path: Path, feature_names=None, details_path: Path = None):
        PathBuilder.build(path)
        file_path = path / f"{self._get_model_filename()}.pickle"
        with file_path.open("wb") as file:
            dill.dump(self.model, file)

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        with params_path.open("w") as file:
            desc = {
                **(self.get_params()),
                "feature_names": feature_names,
                "classes": self.model.classes_.tolist(),
                "class_mapping": self.class_mapping,
            }

            if self.label is not None:
                desc["label"] = vars(self.label)

            yaml.dump(desc, file)

    def _get_model_filename(self):
        return FilenameHandler.get_filename(self.__class__.__name__, "")

    def load(self, path: Path, details_path: Path = None):
        name = f"{self._get_model_filename()}.pickle"
        file_path = path / name
        if file_path.is_file():
            with file_path.open("rb") as file:
                self.model = dill.load(file)
        else:
            raise FileNotFoundError(f"{self.__class__.__name__} model could not be loaded from {file_path}"
                                    f". Check if the path to the {name} file is properly set.")

        if details_path is None:
            params_path = path / f"{self._get_model_filename()}.yaml"
        else:
            params_path = details_path

        if params_path.is_file():
            with params_path.open("r") as file:
                desc = yaml.safe_load(file)
                if "label" in desc:
                    setattr(self, "label", Label(**desc["label"]))
                for param in ["feature_names", "classes", "class_mapping"]:
                    if param in desc:
                        setattr(self, param, desc[param])

    def check_if_exists(self, path: Path):
        file_path = path / f"{self._get_model_filename()}.pickle"
        return file_path.is_file()

    @abc.abstractmethod
    def _get_ml_model(self):
        pass

    @abc.abstractmethod
    def get_params(self):
        '''Returns the model parameters in a readable yaml-friendly way (consisting of lists, dictionaries and strings).'''
        pass

    def get_label_name(self):
        return self.label.name

    def get_package_info(self) -> str:
        return 'scikit-learn ' + pkg_resources.get_distribution('scikit-learn').version

    def get_feature_names(self) -> list:
        return self.feature_names

    def get_class_mapping(self) -> dict:
        """Returns a dictionary containing the mapping between label values and values internally used in the classifier"""
        return self.class_mapping

    def get_compatible_encoders(self):
        from immuneML.encodings.evenness_profile.EvennessProfileEncoder import EvennessProfileEncoder
        from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
        from immuneML.encodings.onehot.OneHotEncoder import OneHotEncoder
        from immuneML.encodings.word2vec.Word2VecEncoder import Word2VecEncoder
        from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
        from immuneML.encodings.reference_encoding.MatchedReceptorsEncoder import MatchedReceptorsEncoder
        from immuneML.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder

        return [KmerFrequencyEncoder, OneHotEncoder, Word2VecEncoder, EvennessProfileEncoder,
                MatchedSequencesEncoder, MatchedReceptorsEncoder, MatchedRegexEncoder]

    @staticmethod
    def get_usage_documentation(model_name):
        return f"""
        
        TODO
        
        Following text does not relate to generative models

        Scikit-learn models can be trained in two modes: 

        1. Creating a model using a given set of hyperparameters, and relying on the selection and assessment loop in the
        TrainMLModel instruction to select the optimal model. 

        2. Passing a range of different hyperparameters to {model_name}, and using a third layer of nested cross-validation 
        to find the optimal hyperparameters through grid search. In this case, only the {model_name} model with the optimal 
        hyperparameter settings is further used in the inner selection loop of the TrainMLModel instruction. 

        By default, mode 1 is used. In order to use mode 2, model_selection_cv and model_selection_n_folds must be set. 


        Arguments:

            {model_name} (dict): Under this key, hyperparameters can be specified that will be passed to the scikit-learn class.
            Any scikit-learn hyperparameters can be specified here. In mode 1, a single value must be specified for each of the scikit-learn
            hyperparameters. In mode 2, it is possible to specify a range of different hyperparameters values in a list. It is also allowed
            to mix lists and single values in mode 2, in which case the grid search will only be done for the lists, while the
            single-value hyperparameters will be fixed. 
            In addition to the scikit-learn hyperparameters, parameter show_warnings (True/False) can be specified here. This determines
            whether scikit-learn warnings, such as convergence warnings, should be printed. By default show_warnings is True.

            model_selection_cv (bool): If any of the hyperparameters under {model_name} is a list and model_selection_cv is True, 
            a grid search will be done over the given hyperparameters, using the number of folds specified in model_selection_n_folds.
            By default, model_selection_cv is False. 

            model_selection_n_folds (int): The number of folds that should be used for the cross validation grid search if model_selection_cv is True.

            """


