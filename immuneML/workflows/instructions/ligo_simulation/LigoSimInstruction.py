import copy
import dataclasses
import math
import random
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from bionumpy.bnpdataclass import BNPDataClass

from immuneML.IO.dataset_export.DataExporter import DataExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LigoSimState import LigoSimState
from immuneML.simulation.SimConfig import SimConfig
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.simulation_strategy.ImplantingStrategy import ImplantingStrategy
from immuneML.simulation.util.bnp_util import merge_dataclass_objects
from immuneML.simulation.util.util import get_bnp_data, make_receptor_sequence_objects, make_annotated_dataclass, get_sequence_per_signal_count, \
    update_seqs_without_signal, update_seqs_with_signal, check_iteration_progress, make_sequence_paths, make_signal_metadata, needs_seqs_with_signal, \
    get_signal_sequence_count, check_sequence_count, make_repertoire_from_sequences, get_no_signal_sequences, get_signal_sequences, \
    annotate_sequences
from immuneML.util.ExporterHelper import ExporterHelper
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.Instruction import Instruction


class LigoSimInstruction(Instruction):
    """
    LIgO simulation instruction creates a synthetic dataset from scratch based on the generative model and a set of signals provided by
    the user.

    Arguments:

        simulation (str): a name of a simulation object containing a list of SimConfigItem as specified under definitions key; defines how
        to combine signals with simulated data; specified under definitions

        store_signal_in_receptors (bool): for repertoire-level simulation, whether to store the information on what exact motif is implanted in each receptor

        sequence_batch_size (bool): how many sequences to generate at once using the generative model before checking for signals and filtering

        max_iterations (int): how many iterations are allowed when creating sequences

        export_p_gens (bool): whether to compute generation probabilities (if supported by the generative model) for sequences and include them as part of output

        number_of_processes (int): determines how many simulation items can be simulated in parallel

        export_formats: in which formats to export the dataset after simulation. Valid formats are class names of any non-abstract class
        inheriting :py:obj:`~immuneML.IO.dataset_export.DataExporter.DataExporter`. Important note: Binary files in ImmuneML might not be compatible
        between different immuneML versions.

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_simulation_instruction: # user-defined name of the instruction
            type: LIgOSimulation # which instruction to execute
            simulation: sim1
            store_signal_in_receptors: True
            sequence_batch_size: 1000
            max_iterations: 1000
            export_p_gens: False
            number_of_processes: 4
            export_formats: [AIRR] # in which formats to export the dataset

    """

    def __init__(self, simulation: SimConfig, signals: List[Signal], name: str, store_signal_in_receptors: bool,
                 sequence_batch_size: int, max_iterations: int, number_of_processes: int, exporters: List[DataExporter] = None,
                 export_p_gens: bool = None):

        self.state = LigoSimState(simulation=simulation, signals=signals, name=name, store_signal_in_receptors=store_signal_in_receptors)
        self._number_of_processes = number_of_processes
        self._sequence_batch_size = sequence_batch_size
        self._max_iterations = max_iterations
        self._exporters = exporters
        self._export_p_gens = export_p_gens

        self._use_p_gens = (self._export_p_gens or self.state.simulation.keep_p_gen_dist) and \
                           all(sim_item.generative_model.can_compute_p_gens() for sim_item in self.state.simulation.sim_items)

        self._export_observed_signals = any([it.false_negative_prob_in_receptors > 0 or it.false_positive_prob_in_receptors > 0
                                             for it in self.state.simulation.sim_items])

        self._noise_fields = [(f"observed_{s.id}", int) for s in self.state.signals] if self._export_observed_signals else []

        self._annotation_fields = sorted([(signal.id, int) for signal in self.state.signals] +
                                         [(f"{signal.id}_positions", str) for signal in self.state.signals] + self._noise_fields, key=lambda x: x[0])

        self._custom_fields = self._annotation_fields + [('p_gen', float)]
        self._background_fields = [(field.name, field.type) for field in dataclasses.fields(BackgroundSequences)]

    @property
    def sequence_type(self) -> SequenceType:
        return self.state.simulation.sequence_type

    @property
    def _annotated_dataclass(self):
        return make_annotated_dataclass(self._annotation_fields, self.state.signals)

    MIN_RANGE_PROBABILITY = 1e-5

    def run(self, result_path: Path):
        self.state.result_path = PathBuilder.build(result_path / self.state.name)

        self._simulate_dataset()
        self._export_dataset()

        return self.state

    def _export_dataset(self):
        exporter_output = ExporterHelper.export_dataset(self.state.resulting_dataset, self._exporters, self.state.result_path)

        self.state.formats = exporter_output['formats']
        self.state.paths = exporter_output['paths']

    def _simulate_dataset(self):
        chunk_size = math.ceil(len(self.state.simulation.sim_items) / self._number_of_processes)

        with Pool(processes=max(self._number_of_processes, len(self.state.simulation.sim_items))) as pool:
            result = pool.map(self._create_examples, [vars(item) for item in self.state.simulation.sim_items], chunksize=chunk_size)
            examples = list(chain.from_iterable(result))
            random.shuffle(examples)

        labels = {signal.id: [True, False] for signal in self.state.signals}

        if self.state.simulation.is_repertoire:
            self.state.resulting_dataset = RepertoireDataset.build_from_objects(labels=labels, repertoires=examples, name='simulated_dataset',
                                                                                metadata_path=self.state.result_path / 'metadata.csv')
        elif self.state.simulation.paired:
            raise NotImplementedError()
        else:
            self.state.resulting_dataset = SequenceDataset.build_from_objects(examples, path=self.state.result_path, name='simulated_dataset',
                                                                              file_size=SequenceDataset.DEFAULT_FILE_SIZE, labels=labels)

    def _create_examples(self, item_in: dict):

        item = SimConfigItem(**item_in)

        if self.state.simulation.is_repertoire:
            res = self._create_repertoires(item)
        else:
            res = self._create_receptors(item)

        return res

    def _create_receptors(self, sim_item: SimConfigItem) -> list:
        if self.state.simulation.paired:
            raise NotImplementedError

        assert len(sim_item.signals) in [0, 1], f"{LigoSimInstruction.__name__}: for sequence datasets, only 0 or 1 signal per sequence are " \
                                                f"supported, but {len(sim_item.signals)} were specified."

        sequence_paths = self._gen_necessary_sequences(self.state.result_path, sim_item)
        sequences = None

        if len(sim_item.signals) == 0:
            sequences = get_bnp_data(sequence_paths['no_signal'], self._annotated_dataclass)
        else:
            for signal in sim_item.signals:
                signal_sequences = get_bnp_data(sequence_paths[signal.id], self._annotated_dataclass)
                sequences = signal_sequences if sequences is None else merge_dataclass_objects([sequences, signal_sequences])

        sequences = make_receptor_sequence_objects(sequences, metadata=make_signal_metadata(sim_item, self.state.signals),
                                                   immune_events=sim_item.immune_events, custom_params=self._custom_fields)

        return sequences

    def _create_repertoires(self, item: SimConfigItem) -> list:
        path = PathBuilder.build(self.state.result_path / item.name)
        sequence_paths = self._gen_necessary_sequences(path, sim_item=item)

        repertoires = []
        used_seq_count = {**{'no_signal': 0}, **{signal.id: 0 for signal in item.signals}}
        repertoires_path = PathBuilder.build(path / "repertoires")

        for i in range(item.number_of_examples):
            seqs_no_signal_count = item.receptors_in_repertoire_count - get_signal_sequence_count(repertoire_count=1, sim_item=item) * len(
                item.signals)

            sequences, used_seq_count = get_no_signal_sequences(used_seq_count=used_seq_count, seqs_no_signal_count=seqs_no_signal_count,
                                                                bnp_data_class=self._annotated_dataclass, sequence_paths=sequence_paths)

            sequences, used_seq_count = get_signal_sequences(sequences, self._annotated_dataclass, used_seq_count, item, sequence_paths)

            check_sequence_count(item, sequences)

            repertoire = make_repertoire_from_sequences(sequences, repertoires_path, item, self.state.signals, self._custom_fields)
            repertoires.append(repertoire)

        return repertoires

    def _gen_necessary_sequences(self, base_path: Path, sim_item: SimConfigItem) -> Dict[str, Path]:
        path = PathBuilder.build(base_path)
        seqs_per_signal_count = get_sequence_per_signal_count(sim_item)
        seq_paths = make_sequence_paths(path, sim_item.signals)
        iteration = 0

        while sum(seqs_per_signal_count.values()) > 0 and iteration < self._max_iterations:
            sequences = self._make_background_sequences(path, iteration, sim_item, seqs_per_signal_count)

            if self.state.simulation.keep_p_gen_dist and iteration == 0:
                self._make_p_gen_histogram(sequences)

            sequences = annotate_sequences(sequences, self.sequence_type == SequenceType.AMINO_ACID, self.state.signals, self._annotated_dataclass)

            sequences = self.state.simulation.simulation_strategy.process_sequences(sequences, copy.deepcopy(seqs_per_signal_count), self._use_p_gens,
                                                                                    self.sequence_type, sim_item, self.state.signals,
                                                                                    self.state.simulation.remove_seqs_with_signals)

            if self.state.simulation.keep_p_gen_dist:
                sequences = self._filter_using_p_gens(sequences, sim_item)

            seqs_per_signal_count['no_signal'] = update_seqs_without_signal(seqs_per_signal_count['no_signal'], sequences, seq_paths['no_signal'])
            seqs_per_signal_count = update_seqs_with_signal(copy.deepcopy(seqs_per_signal_count), sequences, self.state.signals, sim_item.signals,
                                                            seq_paths)

            check_iteration_progress(iteration, self._max_iterations)
            iteration += 1

        if iteration == self._max_iterations and sum(seqs_per_signal_count.values()) != 0:
            raise RuntimeError(f"{LigoSimInstruction.__name__}: maximum iterations were reached, but the simulation could not finish "
                               f"with parameters: {vars(self.state.simulation)}.\n")

        return seq_paths

    def _make_background_sequences(self, path, iteration: int, sim_item: SimConfigItem, sequence_per_signal_count: dict) -> BackgroundSequences:
        sequence_path = PathBuilder.build(path / f"gen_model/") / f"tmp_{iteration}.tsv"

        sim_item.generative_model.generate_sequences(self._sequence_batch_size, seed=sim_item.seed, path=sequence_path,
                                                     sequence_type=self.sequence_type, compute_p_gen=self._use_p_gens)

        v_genes = sorted(
            list(set(chain.from_iterable([[motif.v_call for motif in signal.motifs if motif.v_call] for signal in sim_item.signals]))))
        j_genes = sorted(
            list(set(chain.from_iterable([[motif.j_call for motif in signal.motifs if motif.j_call] for signal in sim_item.signals]))))

        skew_model_for_signal = needs_seqs_with_signal(sequence_per_signal_count)

        if sim_item.generative_model.can_generate_from_skewed_gene_models() and skew_model_for_signal and (len(v_genes) > 0 or len(j_genes) > 0):
            sim_item.generative_model.generate_from_skewed_gene_models(v_genes=v_genes, j_genes=j_genes, seed=sim_item.seed, path=sequence_path,
                                                                       sequence_type=self.sequence_type, batch_size=self._sequence_batch_size,
                                                                       compute_p_gen=self._use_p_gens)

        return get_bnp_data(sequence_path, BackgroundSequences)

    def _make_p_gen_histogram(self, sequences: BackgroundSequences):

        log_p_gens = np.log10(sequences.p_gen)
        hist, self.state.p_gen_bins = np.histogram(log_p_gens, density=False, bins=np.concatenate(
            ([np.NINF], np.histogram_bin_edges(log_p_gens, self.state.simulation.p_gen_bin_count), [np.PINF])))
        self.state.target_p_gen_histogram = hist / len(sequences)

        zero_regions = self.state.target_p_gen_histogram == 0
        self.state.target_p_gen_histogram[zero_regions] = ImplantingStrategy.MIN_RANGE_PROBABILITY
        self.state.target_p_gen_histogram[np.logical_not(zero_regions)] -= \
            ImplantingStrategy.MIN_RANGE_PROBABILITY * np.sum(zero_regions) / np.sum(np.logical_not(zero_regions))

    def _filter_using_p_gens(self, sequences: BackgroundSequences, sim_item: SimConfigItem) -> Tuple[BNPDataClass, dict]:
        if np.any(sequences.p_gen == -1):
            missing_p_gens = sequences.p_gen == -1
            sequences.p_gen[missing_p_gens] = sim_item.generative_model.compute_p_gens(sequences[missing_p_gens], self.state.simulation.sequence_type)

        p_gens = np.log10(sequences.p_gen)
        sequence_bins = np.digitize(p_gens, self.state.p_gen_bins) - 1
        keep_sequences = np.random.uniform(0, 1, len(sequences)) <= self.state.target_p_gen_histogram[sequence_bins]

        return sequences[keep_sequences]
