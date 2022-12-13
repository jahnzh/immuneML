import os
import shutil
from unittest import TestCase

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.MotifGeneralizationAnalysis import MotifGeneralizationAnalysis
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestMotifGeneralizationAnalysis(TestCase):
    def test_generate(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "significant_motif_overlap/")


        dataset = RandomDatasetGenerator.generate_sequence_dataset(100, {10: 1}, {"l1": {"A": 0.5, "B": 0.5}}, path / "dataset")


        identifiers = [seq.identifier for seq in dataset.get_data()]
        training_set_identifiers = identifiers[::2]

        with open(path / "training_ids.txt", "w") as identifiers_file:
            identifiers_file.writelines([identifier + "\n" for identifier in training_set_identifiers])

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "MotifGeneralizationAnalysis")
        params["training_set_identifier_path"] = str(path / "training_ids.txt")
        params["max_positions"] = 1
        params["min_precision"] = 0.8
        params["random_seed"] = 1
        params["min_points_in_window"] = 2
        params["dataset"] = dataset
        params["result_path"] = path / "result"
        params["label"] = {"l1": {"positive_class": "A"}}

        report = MotifGeneralizationAnalysis.build_object(**params)

        report._generate()


        self.assertTrue(os.path.isdir(path / "result/datasets/train"))
        self.assertTrue(os.path.isdir(path / "result/datasets/test"))
        self.assertTrue(os.path.isdir(path / "result/encoded_data"))

        self.assertTrue(os.path.isfile(path / "result/training_set_scores.csv"))
        self.assertTrue(os.path.isfile(path / "result/test_set_scores.csv"))
        self.assertTrue(os.path.isfile(path / "result/training_combined_precision.csv"))
        self.assertTrue(os.path.isfile(path / "result/test_combined_precision.csv"))

        self.assertTrue(os.path.isfile(path / "result/training_precision_per_tp.html"))
        self.assertTrue(os.path.isfile(path / "result/test_precision_per_tp.html"))

        self.assertTrue(os.path.isfile(path / "result/training_precision_recall.html"))
        self.assertTrue(os.path.isfile(path / "result/test_precision_recall.html"))

        shutil.rmtree(path)
