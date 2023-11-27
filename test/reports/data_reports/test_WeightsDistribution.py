import plotly.express as px
import warnings
from pathlib import Path

import pandas as pd

from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.reports.ReportOutput import ReportOutput
from immuneML.reports.ReportResult import ReportResult
from immuneML.reports.data_reports.DataReport import DataReport
from immuneML.util.PathBuilder import PathBuilder
from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper

import os
import shutil
import pandas as pd
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.motif_encoding.MotifEncoder import MotifEncoder
from immuneML.reports.data_reports.WeightsDistribution import WeightsDistribution
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.ReportResult import ReportResult
from immuneML.util.PathBuilder import PathBuilder

class TestWeightsDistribution(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _create_dummy_encoded_data(self, path):
        sequences = [
            ReceptorSequence(
                sequence_aa="AACC",
                sequence_id="1",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                sequence_aa="AGDD",
                sequence_id="2",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                sequence_aa="AAEE",
                sequence_id="3",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                sequence_aa="AGFF",
                sequence_id="4",
                metadata=SequenceMetadata(custom_params={"l1": 1}),
            ),
            ReceptorSequence(
                sequence_aa="CCCC",
                sequence_id="5",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
            ReceptorSequence(
                sequence_aa="DDDD",
                sequence_id="6",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
            ReceptorSequence(
                sequence_aa="EEEE",
                sequence_id="7",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
            ReceptorSequence(
                sequence_aa="FFFF",
                sequence_id="8",
                metadata=SequenceMetadata(custom_params={"l1": 2}),
            ),
        ]

        PathBuilder.build(path)

        dataset = SequenceDataset.build_from_objects(
            sequences, 100, PathBuilder.build(path / "data"), "d1"
        )

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2], positive_class=1)

        dataset.set_example_weights([i+1 for i in range(dataset.get_example_count())])


        return dataset

    def test_generate(self):
        path = EnvironmentSettings.tmp_test_path / "weight_distribution/"
        PathBuilder.build(path)

        encoded_dataset = self._create_dummy_encoded_data(path)

        label = "is_binding"
        weight_thresholds = [0.001, 0.01, 0.1]
        split_classes = True

        report = WeightsDistribution.build_object(
            **{"dataset": encoded_dataset, "result_path": path, "label": "l1"}
        )

        self.assertTrue(report.check_prerequisites())

        result = report._generate()

        self.assertIsInstance(result, ReportResult)

        # self.assertEqual(result.output_figures[0].path, path / "gap_size_for_motif_size_2.html")

        # content = pd.read_csv(path / "gap_size_table_motif_size_2.csv")
        # self.assertEqual((list(content.columns))[1], "Gap size, occurrence")
        #
        # content = pd.read_csv(path / "positional_aa_counts.csv")
        # self.assertEqual(list(content.index), [i for i in range(4)])

        shutil.rmtree(path)