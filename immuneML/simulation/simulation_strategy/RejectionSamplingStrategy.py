import logging
from typing import List

from bionumpy.bnpdataclass import BNPDataClass

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy
from immuneML.simulation.util.util import filter_out_illegal_sequences


class RejectionSamplingStrategy(SimulationStrategy):

    def process_sequences(self, sequences: BNPDataClass, seqs_per_signal_count: dict, use_p_gens: bool, sequence_type: SequenceType,
                          sim_item: SimConfigItem, all_signals: List[Signal], remove_positives_first: bool) -> BNPDataClass:
        filtered_sequences = filter_out_illegal_sequences(sequences, sim_item, all_signals, 1)

        removed_count = len(sequences) - len(filtered_sequences)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} out of {len(sequences)} during rejection sampling for having more than 1 signal.", True)

        return filtered_sequences
