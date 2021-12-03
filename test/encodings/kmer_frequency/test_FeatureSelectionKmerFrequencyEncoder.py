import os
import shutil
from unittest import TestCase

import numpy

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.ReceptorDataset import ReceptorDataset
from immuneML.data_model.receptor.TCABReceptor import TCABReceptor
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.FeatureSelectionKmerFrequencyEncoder import FeatureSelectionKmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder


class TestKmerFreqReceptorEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def make_receptor_dataset(self, path):
        receptors = [
            TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAA"),
                         beta=ReceptorSequence(amino_acid_sequence="AAA"), identifier="1", metadata={'l1': 1}),
            TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAA"),
                         beta=ReceptorSequence(amino_acid_sequence="CCC"), identifier="2", metadata={'l1': 2}),
            TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAA"),
                         beta=ReceptorSequence(amino_acid_sequence="AAA"), identifier="3", metadata={'l1': 1}),
            TCABReceptor(alpha=ReceptorSequence(amino_acid_sequence="AAA"),
                         beta=ReceptorSequence(amino_acid_sequence="CCC"), identifier="4", metadata={'l1': 2})]

        PathBuilder.build(path / 'data')
        dataset = ReceptorDataset.build_from_objects(receptors, path=path, file_size=10)

        return dataset

    def test_encode(self):
        path = EnvironmentSettings.tmp_test_path / "feature_sel_kmer_rec_freq_p_value/"
        dataset = self.make_receptor_dataset(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        encoder = FeatureSelectionKmerFrequencyEncoder.build_object(dataset, **{
            'kmer_encoder': {
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                "sequence_type": SequenceType.AMINO_ACID.name,
                "k": 3
            },
            'p_value_threshold': 0.05,
            'alternative_hypothesis': 'two-sided',
            'equal_variance': True,
            'top_n_percent_features': 0.01,
            'name': 'enc1'
        })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "2/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv",
            encode_labels=True
        ))

        self.assertEqual(4, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4']))
        self.assertTrue(numpy.array_equal(encoded_dataset.encoded_data.examples[0].A, encoded_dataset.encoded_data.examples[2].A))
        self.assertTrue(all(feature_name in encoded_dataset.encoded_data.feature_names for feature_name in ["beta_AAA", "beta_CCC"]))

        shutil.rmtree(path)

    def test_encode_top_n_features(self):
        path = EnvironmentSettings.tmp_test_path / "feature_sel_kmer_rec_freq_top_n_features/"
        dataset = self.make_receptor_dataset(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        encoder = FeatureSelectionKmerFrequencyEncoder.build_object(dataset, **{
            'kmer_encoder': {
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                "sequence_type": SequenceType.AMINO_ACID.name,
                "k": 3
            },
            'p_value_threshold': None,
            'top_n_percent_features': 0.5,
            'alternative_hypothesis': 'two-sided',
            'equal_variance': True,
            'name': 'enc1'
        })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "2/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv",
            encode_labels=True
        ))

        self.assertEqual(4, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4']))
        self.assertTrue(numpy.array_equal(encoded_dataset.encoded_data.examples[0].A, encoded_dataset.encoded_data.examples[2].A))
        self.assertTrue(all(feature_name in encoded_dataset.encoded_data.feature_names for feature_name in ["beta_AAA"]))

        shutil.rmtree(path)
