import copy
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

        kmer_params (dict): a set of parameters to be used with KmerFrequency encoder, as the first step of this encoding. For the full list of
        parameters, see :ref:`KmerFrequency` encoder documentation.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

            my_selected_kmers:
                FeatureSelectionKmerFrequency:
                    p_value_threshold: 0.05
                    equal_variance: True
                    alternative_hypothesis: two-sided
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

        ParameterValidator.assert_keys(params.keys(), ['p_value_threshold', 'alternative_hypothesis', 'equal_variance', 'kmer_params', 'name'],
                                       location, location)

        ParameterValidator.assert_type_and_value(params['p_value_threshold'], float, location, 'p_value_threshold', 0., 1.)
        ParameterValidator.assert_type_and_value(params['alternative_hypothesis'], str, location, 'alternative_hypothesis')
        ParameterValidator.assert_in_valid_list(params['alternative_hypothesis'], ['two-sided', 'less', 'greater'], location, 'alternative_hypothesis')
        ParameterValidator.assert_type_and_value(params['equal_variance'], bool, location, 'equal_variance')

        kmer_encoder = KmerFrequencyEncoder.build_object(dataset, **params['kmer_params'])

        return FeatureSelectionKmerFrequencyEncoder(p_value_threshold=params['p_value_threshold'], equal_variance=params['equal_variance'],
                                                    alternative_hypothesis=params['alternative_hypothesis'], kmer_encoder=kmer_encoder)

    def __init__(self, p_value_threshold: float, alternative_hypothesis: str, equal_variance: bool, kmer_encoder: KmerFrequencyEncoder):
        self.p_value_threshold = p_value_threshold
        self.alternative_hypothesis = alternative_hypothesis
        self.equal_variance = equal_variance
        self.kmer_encoder = kmer_encoder
        self.features = []

    def encode(self, dataset: Dataset, params: EncoderParams):

        label = self._prepare_label(params)

        kmer_params = copy.deepcopy(params)
        kmer_params.result_path = params.result_path / "kmer_frequency_step"

        kmer_dataset = self.kmer_encoder.encode(dataset, kmer_params)

        encoded_dataset = self._filter_features(kmer_dataset, label)

        return encoded_dataset

    def _filter_features(self, dataset: Dataset, label: Label) -> Dataset:

        encoded_data = dataset.encoded_data
        label_array = np.array(encoded_data.labels[label.name])

        class1_selection = label_array == label.positive_class
        class2_selection = np.logical_not(class1_selection)

        _, p_values = ttest_ind(encoded_data.examples[class1_selection, :].todense(), encoded_data.examples[class2_selection, :].todense(),
                                equal_var=self.equal_variance, alternative=self.alternative_hypothesis)

        feature_indices = p_values < self.p_value_threshold
        self.features = np.array(encoded_data.feature_names)[feature_indices].tolist()

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data.examples = encoded_data.examples[:, feature_indices]

        return encoded_dataset

    def  _prepare_label(self, params: EncoderParams):

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
