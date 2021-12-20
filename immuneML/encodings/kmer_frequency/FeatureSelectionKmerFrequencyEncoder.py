import copy
import logging
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.environment.Label import Label
from immuneML.util.ParameterValidator import ParameterValidator


class FeatureSelectionKmerFrequencyEncoder(DatasetEncoder):
    """
    This encoder represents the dataset in terms of k-mer frequencies and then performs feature selection using t-test. For k-mer frequency encoding,
    under the hood it uses the KmerFrequency encoder and then uses
    `scipy's t-test function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_ for feature selection. It works only
    when the specified label has 2 classes.

    Note: for this encoder to work, it is necessary to set "positive_class" attribute of the label in the YAML specification. It will always be used
    to construct the sample a from t-test (see scipy's t-test documentaation).

    Arguments:

        p_value_threshold (float): p-value threshold for keeping features after the t-test

        alternative_hypothesis (str): parameter of scipy's t-test function; values: `two-sided`, `less`, or `greater`

        equal_variance (bool): parameter of scipy's t-test function; if true, t-test is used, if false, Welch's t-test is used which does not assume
        equal population variance

        top_n_percent_features (float): alternatively to selecting features based on the p-value threshold, this parameter allows to select top n
        percent of features that have the smallest p-value; this should be a value between 0 and 1; if both p-value threshold and
        top_n_percent_features are specified at the same time, p-value threshold will be attempted first and if no features are selected,
        top_n_percent_features parameter will be used instead

        kmer_encoder (dict): a set of parameters to be used with KmerFrequency encoder, as the first step of this encoding. For the full list of
        parameters, see :ref:`KmerFrequency` encoder documentation.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_selected_kmers:
                FeatureSelectionKmerFrequency:
                    p_value_threshold: 0.05
                    equal_variance: True
                    alternative_hypothesis: two-sided
                    top_n_percent_features: 0.01
                    kmer_params:
                        normalization_type: RELATIVE_FREQUENCY
                        reads: UNIQUE
                        sequence_encoding: CONTINUOUS_KMER
                        sequence_type: AMINO_ACID
                        k: 3
                        scale_to_unit_variance: True
                        scale_to_zero_mean: True

    """

    @staticmethod
    def build_object(dataset, **params):

        location = FeatureSelectionKmerFrequencyEncoder.__name__

        ParameterValidator.assert_keys(params.keys(), ['p_value_threshold', 'alternative_hypothesis', 'equal_variance', 'top_n_percent_features',
                                                       'kmer_encoder', 'name'], location, location)

        if params['p_value_threshold']:
            ParameterValidator.assert_type_and_value(params['p_value_threshold'], float, location, 'p_value_threshold', 0., 1.)
        ParameterValidator.assert_type_and_value(params['alternative_hypothesis'], str, location, 'alternative_hypothesis')
        ParameterValidator.assert_in_valid_list(params['alternative_hypothesis'], ['two-sided', 'less', 'greater'], location,
                                                'alternative_hypothesis')
        ParameterValidator.assert_type_and_value(params['top_n_percent_features'], float, location, 'top_n_percent_features', 0., 1.)
        ParameterValidator.assert_type_and_value(params['equal_variance'], bool, location, 'equal_variance')

        kmer_encoder = KmerFrequencyEncoder.build_object(dataset, **params['kmer_encoder'])

        return FeatureSelectionKmerFrequencyEncoder(p_value_threshold=params['p_value_threshold'], equal_variance=params['equal_variance'],
                                                    alternative_hypothesis=params['alternative_hypothesis'], kmer_encoder=kmer_encoder,
                                                    name=params['name'], top_n_percent_features=params['top_n_percent_features'])

    def __init__(self, p_value_threshold: float, alternative_hypothesis: str, equal_variance: bool, top_n_percent_features: float,
                 kmer_encoder: KmerFrequencyEncoder, name: str):
        self.p_value_threshold = p_value_threshold
        self.alternative_hypothesis = alternative_hypothesis
        self.equal_variance = equal_variance
        self.top_n_percent_features = top_n_percent_features
        self.kmer_encoder = kmer_encoder
        self.name = name
        self.features = []

    def encode(self, dataset: Dataset, params: EncoderParams):

        label = self._prepare_label(params)

        kmer_params = copy.deepcopy(params)
        kmer_params.result_path = params.result_path / "kmer_frequency_step"

        kmer_dataset = self.kmer_encoder.encode(dataset, kmer_params)

        encoded_dataset = self._filter_features(kmer_dataset, label, params.learn_model)

        return encoded_dataset

    def _filter_features(self, dataset: Dataset, label: Label, learn_model: bool) -> Dataset:

        encoded_data = dataset.encoded_data

        if learn_model:

            class1_data, class2_data = self._get_per_class_data(encoded_data, label)

            _, p_values = ttest_ind(class1_data, class2_data, equal_var=self.equal_variance, alternative=self.alternative_hypothesis)

            if self.p_value_threshold is not None:
                feature_indices = p_values.flatten() < self.p_value_threshold
                if feature_indices.sum() == 0:
                    logging.warning(f"{FeatureSelectionKmerFrequencyEncoder.__name__}: no features were selected as relevant in encoder {self.name}. "
                                    f"Try adjusting the parameters of the encoding. Using all features for now.")
                    feature_indices = self._get_top_n_percent_indices(p_values.flatten())
            else:
                feature_indices = self._get_top_n_percent_indices(p_values.flatten())
            self.features = np.array(encoded_data.feature_names)[feature_indices].tolist()
        else:
            feature_indices = [i for i in range(len(encoded_data.feature_names)) if encoded_data.feature_names[i] in self.features]

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data.examples = encoded_data.examples[:, feature_indices]
        encoded_dataset.encoded_data.feature_names = self.features

        return encoded_dataset

    def _get_top_n_percent_indices(self, p_values):
        feature_count_to_keep = int(p_values.shape[0] * self.top_n_percent_features)
        if feature_count_to_keep == 0:
            logging.warning(f"{FeatureSelectionKmerFrequencyEncoder.__name__}: no features could be selected in {self.name} "
                            f"encoder, keeping all features...")
            return np.ones_like(p_values, dtype=bool)
        else:
            indices_to_keep = np.argsort(p_values)[:feature_count_to_keep].astype(int)
            return np.array([True if index in indices_to_keep else False for index in range(p_values.shape[0])])

    def _get_per_class_data(self, encoded_data, label: Label):
        label_array = np.array(encoded_data.labels[label.name])

        class1_selection = label_array == label.positive_class
        class2_selection = np.logical_not(class1_selection)

        if isinstance(encoded_data.examples, np.ndarray):
            class1_data = encoded_data.examples[class1_selection, :]
            class2_data = encoded_data.examples[class2_selection, :]
        else:
            class1_data = encoded_data.examples[class1_selection, :].todense()
            class2_data = encoded_data.examples[class2_selection, :].todense()

        return class1_data, class2_data

    def _prepare_label(self, params: EncoderParams):

        assert len(params.label_config.get_label_objects()) == 1, f"{FeatureSelectionKmerFrequencyEncoder.__name__}: no label object provided."

        label = params.label_config.get_label_objects()[0]

        assert label.positive_class, f"{FeatureSelectionKmerFrequencyEncoder.__name__}: label {label.name} has positive class set to " \
                                     f"{label.positive_class}. To use this encoder, set the positive class."

        return label

    @staticmethod
    def export_encoder(path: Path, encoder) -> Path:
        encoder_file = DatasetEncoder.store_encoder(encoder, path / "encoder.pickle")
        return encoder_file

    @staticmethod
    def load_encoder(encoder_file: Path):
        encoder = DatasetEncoder.load_encoder(encoder_file)
        return encoder
