import dataclasses
import logging
import uuid
from dataclasses import make_dataclass
from itertools import chain
from pathlib import Path
from typing import List

import bionumpy as bnp
import numpy as np
from bionumpy import AminoAcidEncoding, DNAEncoding, EncodedRaggedArray
from bionumpy.bnpdataclass import bnpdataclass, BNPDataClass
from bionumpy.encodings import BaseEncoding
from bionumpy.io import delimited_buffers
from bionumpy.sequence.string_matcher import RegexMatcher, StringMatcher
from npstructures import RaggedArray

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.generative_models.GenModelAsTSV import GenModelAsTSV
from immuneML.simulation.implants.MotifInstance import MotifInstance
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PositionHelper import PositionHelper


def get_signal_sequence_count(repertoire_count: int, sim_item) -> int:
    return round(sim_item.receptors_in_repertoire_count * sim_item.repertoire_implanting_rate) * repertoire_count


def get_sequence_per_signal_count(sim_item) -> dict:
    if sim_item.receptors_in_repertoire_count:
        sequence_count = sim_item.receptors_in_repertoire_count * sim_item.number_of_examples
        seq_with_signal_count = {signal.id: get_signal_sequence_count(repertoire_count=sim_item.number_of_examples, sim_item=sim_item)
                                 for signal in sim_item.signals}
        seq_without_signal_count = {'no_signal': sequence_count - sum(seq_with_signal_count.values())}

        return {**seq_with_signal_count, **seq_without_signal_count}
    elif len(sim_item.signals) == 1:
        return {sim_item.signals[0].id: sim_item.number_of_examples, 'no_signal': 0}
    elif len(sim_item.signals) == 0:
        return {'no_signal': sim_item.number_of_examples}
    else:
        raise NotImplementedError


def get_bnp_data(sequence_path, columns_with_types: list = None):
    if columns_with_types is None or isinstance(columns_with_types, list) and len(columns_with_types) == 0:
        data_class = GenModelAsTSV
    else:
        data_class = GenModelAsTSV.extend(tuple(columns_with_types))

    buff_type = delimited_buffers.get_bufferclass_for_datatype(data_class, delimiter='\t', has_header=True)

    with bnp.open(sequence_path, buffer_type=buff_type) as file:
        data = file.read()

    return data


def make_receptor_sequence_objects(sequences: GenModelAsTSV, all_signals, metadata, immune_events: dict) -> List[ReceptorSequence]:
    custom_params = get_custom_keys(all_signals, [('p_gen', float)])

    return [ReceptorSequence(seq.sequence_aa.to_string(), seq.sequence.to_string(), identifier=uuid.uuid4().hex,
                             metadata=construct_sequence_metadata_object(seq, metadata, custom_params, immune_events)) for seq in sequences]


def construct_sequence_metadata_object(sequence, metadata: dict, custom_params, immune_events: dict) -> SequenceMetadata:
    custom = {}

    for key, key_type in custom_params:
        if 'position' in key:
            custom[key] = getattr(sequence, key).to_string()
        else:
            custom[key] = getattr(sequence, key).item()

    return SequenceMetadata(custom_params={**metadata, **custom, **immune_events},
                            v_call=sequence.v_call.to_string(), j_call=sequence.j_call.to_string(), region_type=sequence.region_type.to_string())


def write_bnp_data(path: Path, data, append_if_exists: bool = True):
    buff_type = delimited_buffers.get_bufferclass_for_datatype(type(data), delimiter="\t", has_header=True)

    if path.is_file() and append_if_exists:
        with bnp.open(path, buffer_type=buff_type, mode='a') as file:
            file.write(data)
    elif not path.is_file():
        with bnp.open(path, buffer_type=buff_type, mode='w') as file:
            file.write(data)
    else:
        raise RuntimeError(f"Tried writing to {path}, but it already exists and append_if_exists parameter is set to False.")


def get_allowed_positions(signal: Signal, sequence_array: RaggedArray, region_type: RegionType):
    sequence_lengths = sequence_array.lengths
    if signal.sequence_position_weights is not None:
        signal_positions = [key for key, val in signal.sequence_position_weights.items() if val > 0]
        allowed_positions = RaggedArray([[pos in signal_positions for pos in PositionHelper.gen_imgt_positions_from_length(seq_len, region_type)]
                                         for seq_len in sequence_lengths])
    else:
        allowed_positions = None

    return allowed_positions


def get_region_type(sequences) -> RegionType:
    if hasattr(sequences, "region_type") and \
        np.all([el.to_string() == getattr(sequences, 'region_type')[0].to_string() for el in getattr(sequences, 'region_type')]):
        return RegionType[getattr(sequences, 'region_type')[0].to_string()]
    else:
        raise RuntimeError(f"The region types could not be obtained.")


def annotate_sequences(sequences, is_amino_acid: bool, all_signals: list):
    encoding = AminoAcidEncoding if is_amino_acid else DNAEncoding
    sequence_array = sequences.sequence_aa if is_amino_acid else sequences.sequence
    region_type = get_region_type(sequences)

    signal_matrix = np.zeros((len(sequence_array), len(all_signals)))
    signal_positions = {}

    for index, signal in enumerate(all_signals):
        signal_pos_col = None
        allowed_positions = get_allowed_positions(signal, sequence_array, region_type)

        for motifs, v_call, j_call in signal.get_all_motif_instances(SequenceType.AMINO_ACID if is_amino_acid else SequenceType.NUCLEOTIDE):
            matches_gene = match_genes(v_call, sequences.v_call, j_call, sequences.j_call)
            matches = None

            for motif in motifs:

                matches_motif = match_motif(motif, encoding, sequence_array)
                if matches is None:
                    matches = np.logical_and(matches_motif, matches_gene)
                else:
                    matches = np.logical_or(matches, np.logical_and(matches_motif, matches_gene))

            if allowed_positions is not None:
                matches = np.logical_and(matches, allowed_positions)

            signal_pos_col = np.logical_or(signal_pos_col, matches) if signal_pos_col is not None else matches
            signal_matrix[:, index] = np.logical_or(signal_matrix[:, index], np.logical_or.reduce(matches, axis=1))

        np_mask = RaggedArray(np.where(signal_pos_col.ravel(), "1", "0"), shape=signal_pos_col.shape)
        signal_positions[f'{signal.id}_positions'] = ['m' + "".join(np_mask[ind, :]) for ind in range(len(signal_pos_col))]

    signal_matrix = make_bnp_annotated_sequences(sequences, all_signals, signal_matrix, signal_positions)

    return signal_matrix


def match_genes(v_call, v_call_array, j_call, j_call_array):
    if v_call is not None and v_call != "":
        matcher = StringMatcher(v_call, encoding=BaseEncoding)
        matches_gene = matcher.rolling_window(v_call_array, mode='same').any(axis=1).reshape(-1, 1)
    else:
        matches_gene = np.ones(len(v_call_array)).reshape(-1, 1)

    if j_call is not None and j_call != "":
        matcher = StringMatcher(j_call, encoding=BaseEncoding)
        matches_j = matcher.rolling_window(j_call_array, mode='same').any(axis=1).reshape(-1, 1)
        matches_gene = np.logical_and(matches_gene, matches_j)

    return matches_gene.astype(bool)


def match_motif(motif: str, encoding, sequence_array):
    matcher = RegexMatcher(motif, encoding=encoding)
    matches = matcher.rolling_window(sequence_array, mode='same')
    return matches


def filter_out_illegal_sequences(sequences, sim_item: LIgOSimulationItem, all_signals: list, max_signals_per_sequence: int):
    if max_signals_per_sequence > 1:
        raise NotImplementedError
    elif max_signals_per_sequence == -1:
        return sequences

    sim_signal_ids = [signal.id for signal in sim_item.signals]
    other_signals = [signal.id not in sim_signal_ids for signal in all_signals]
    signal_matrix = sequences.get_signal_matrix()
    legal_indices = np.logical_and(signal_matrix.sum(axis=1) <= max_signals_per_sequence,
                                   np.array(signal_matrix[:, other_signals] == 0).all(axis=1) if any(other_signals) else 1)

    return sequences[legal_indices]


def make_sequences_from_gen_model(sim_item: LIgOSimulationItem, sequence_batch_size: int, seed: int, sequence_path: Path, sequence_type: SequenceType,
                                  skew_model_for_signal: bool):
    sim_item.generative_model.generate_sequences(sequence_batch_size, seed=seed, path=sequence_path, sequence_type=sequence_type)

    if sim_item.generative_model.can_generate_from_skewed_gene_models() and skew_model_for_signal:
        v_genes = sorted(
            list(set(chain.from_iterable([[motif.v_call for motif in signal.motifs if motif.v_call] for signal in sim_item.signals]))))
        j_genes = sorted(
            list(set(chain.from_iterable([[motif.j_call for motif in signal.motifs if motif.j_call] for signal in sim_item.signals]))))

        sim_item.generative_model.generate_from_skewed_gene_models(v_genes=v_genes, j_genes=j_genes, seed=seed, path=sequence_path,
                                                                   sequence_type=sequence_type, batch_size=sequence_batch_size)


def make_bnp_annotated_sequences(sequences: BNPDataClass, all_signals: list, signal_matrix: np.ndarray, signal_positions: dict):
    kwargs = {**{s.id: signal_matrix[:, ind].astype(int) for ind, s in enumerate(all_signals)},
              **{f"{s.id}_positions": bnp.as_encoded_array(signal_positions[f"{s.id}_positions"], bnp.encodings.BaseEncoding) for ind, s in
                 enumerate(all_signals)},
              **{field_name: getattr(sequences, field_name) for field_name in GenModelAsTSV.__annotations__.keys()}}

    bnp_signal_matrix = make_signal_matrix_bnpdataclass(all_signals)(**kwargs)
    return bnp_signal_matrix


def make_signal_matrix_bnpdataclass(signals: list):
    signal_fields = [(s.id, int) for s in signals]
    signal_position_fields = [(f"{s.id}_positions", str) for s in signals]
    base_fields = [(field_name, field_type) for field_name, field_type in GenModelAsTSV.__annotations__.items()]

    functions = {"get_signal_matrix": lambda self: np.array([getattr(self, field) for field, t in signal_fields]).T,
                 "get_signal_names": lambda self: [field for field, t in signal_fields]}

    AnnotatedGenData = bnpdataclass(make_dataclass("AnnotatedGenData", fields=base_fields + signal_fields + signal_position_fields,
                                                   namespace=functions))

    return AnnotatedGenData


def build_imgt_positions(sequence_length: int, motif_instance: MotifInstance, sequence_region_type):
    assert sequence_length >= len(motif_instance), \
        "The motif instance is longer than sequence length. Remove the receptor_sequence from the repertoire or reduce max gap length " \
        "to be able to proceed."

    if sequence_region_type.to_string() == RegionType.IMGT_JUNCTION.name:
        return PositionHelper.gen_imgt_positions_from_junction_length(sequence_length)
    elif sequence_region_type.to_string() == RegionType.IMGT_CDR3.name:
        return PositionHelper.gen_imgt_positions_from_cdr3_length(sequence_length)
    else:
        raise NotImplementedError(f"IMGT positions here are defined only for CDR3 and JUNCTION region types, got {sequence_region_type}")


def choose_implant_position(imgt_positions, position_weights):
    imgt_implant_position = np.random.choice(list(position_weights.keys()), size=1, p=list(position_weights.values()))
    position = np.where(imgt_positions == imgt_implant_position)[0][0]
    return position


def check_iteration_progress(iteration: int, max_iterations: int):
    if iteration == round(max_iterations * 0.75):
        logging.warning(f"Iteration {iteration} out of {max_iterations} max iterations reached during rejection sampling.")


def get_custom_keys(all_signals: List[Signal], custom_keys: list):
    return [(sig.id, int) for sig in all_signals] + [(f'{signal.id}_positions', str) for signal in all_signals] + custom_keys


def check_sequence_count(sim_item, sequences: GenModelAsTSV):
    assert len(sequences) == sim_item.receptors_in_repertoire_count, \
        f"Error when simulating repertoire, needed {sim_item.receptors_in_repertoire_count} sequences, " \
        f"but got {len(sequences)}."


def prepare_data_for_repertoire_obj(all_signals: list, sequences: BNPDataClass, custom_keys: list) -> dict:
    custom_keys = get_custom_keys(all_signals, custom_keys)

    custom_lists = {}
    for field, field_type in custom_keys:
        if field_type is int or field_type is float:
            custom_lists[field] = getattr(sequences, field)
        else:
            custom_lists[field] = [el.to_string() for el in getattr(sequences, field)]

    default_lists = {}
    for field in dataclasses.fields(sequences):
        if field.name not in custom_lists:
            if isinstance(getattr(sequences, field.name), EncodedRaggedArray):
                default_lists[field.name] = [el.to_string() for el in getattr(sequences, field.name)]
            else:
                default_lists[field.name] = getattr(sequences, field.name)

    return {**{"custom_lists": custom_lists}, **default_lists}


def update_seqs_without_signal(max_count, annotated_sequences, seqs_no_signal_path: Path):
    if max_count > 0:
        selection = annotated_sequences.get_signal_matrix().sum(axis=1) == 0
        data_to_write = annotated_sequences[selection][:max_count]
        if len(data_to_write) > 0:
            write_bnp_data(data=data_to_write, path=seqs_no_signal_path)
        return max_count - len(data_to_write)
    else:
        return max_count


def update_seqs_with_signal(max_counts: dict, annotated_sequences, all_signals, sim_item_signals, seqs_with_signal_path: dict):
    all_signal_ids = [signal.id for signal in all_signals]
    signal_matrix = annotated_sequences.get_signal_matrix()

    for signal in sim_item_signals:
        if max_counts[signal.id] > 0:
            selection = signal_matrix[:, all_signal_ids.index(signal.id)].astype(bool)
            data_to_write = annotated_sequences[selection][:max_counts[signal.id]]
            if len(data_to_write) > 0:
                write_bnp_data(data=data_to_write, path=seqs_with_signal_path[signal.id])
            max_counts[signal.id] -= len(data_to_write)

    return max_counts
