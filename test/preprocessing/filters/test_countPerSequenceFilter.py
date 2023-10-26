import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.CountPerSequenceFilter import CountPerSequenceFilter
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestCountPerSequenceFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "count_per_seq_filter/")

        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                         ["ACF", "ACF"],
                                                                         ["ACF", "ACF", "ACF", "ACF"]], path,
                                                                        seq_metadata=[
                                                                            [{"count": 1}, {"count": 2}, {"count": 3}],
                                                                            [{"count": 4}, {"count": 1}],
                                                                            [{"count": 5}, {"count": 6},
                                                                             {"count": None},
                                                                             {"count": 1}]])[0])

        dataset1 = CountPerSequenceFilter(
            **{"low_count_limit": 2, "remove_without_count": True, "remove_empty_repertoires": False,
               "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset1'))
        self.assertEqual(2, dataset1.repertoires[0].get_sequence_aas().shape[0])

        dataset2 = CountPerSequenceFilter(
            **{"low_count_limit": 5, "remove_without_count": True, "remove_empty_repertoires": False,
               "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset2'))
        self.assertEqual(0, dataset2.repertoires[0].get_sequence_aas().shape[0])

        dataset3 = CountPerSequenceFilter(
            **{"low_count_limit": 0, "remove_without_count": True, "remove_empty_repertoires": False,
               "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset3'))
        self.assertEqual(3, dataset3.repertoires[2].get_sequence_aas().shape[0])

        dataset4 = CountPerSequenceFilter(
            **{"low_count_limit": 4, "remove_without_count": True, "remove_empty_repertoires": True,
               "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(
            path / 'with_removed_repertoires'))
        self.assertEqual(2, dataset4.get_example_count())

        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                         ["ACF", "ACF"],
                                                                         ["ACF", "ACF", "ACF", "ACF"]], path,
                                                                        seq_metadata=[[{"count": None}, {"count": None},
                                                                                       {"count": None}],
                                                                                      [{"count": None},
                                                                                       {"count": None}],
                                                                                      [{"count": None}, {"count": None},
                                                                                       {"count": None},
                                                                                       {"count": None}]])[0])

        dataset5 = CountPerSequenceFilter(
            **{"low_count_limit": 0, "remove_without_count": True, "remove_empty_repertoires": False,
               "result_path": path, "batch_size": 4}).process_dataset(dataset, PathBuilder.build(path / 'dataset5'))
        self.assertEqual(0, dataset5.repertoires[0].get_sequence_aas().shape[0])
        self.assertEqual(0, dataset5.repertoires[1].get_sequence_aas().shape[0])
        self.assertEqual(0, dataset5.repertoires[2].get_sequence_aas().shape[0])

        self.assertRaises(AssertionError, CountPerSequenceFilter(**{"low_count_limit": 10, "remove_without_count": True,
                                                               "remove_empty_repertoires": True,
                                                               "result_path": PathBuilder.build(path / 'dataset6'),
                                                               "batch_size": 4}).process_dataset, dataset, path)

        shutil.rmtree(path)
