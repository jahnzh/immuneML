import random
from dataclasses import dataclass
from typing import List, Union

from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.MotifInstance import MotifInstanceGroup
from immuneML.simulation.signal_implanting.SignalImplantingStrategy import SignalImplantingStrategy
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


@dataclass
class Signal:
    """
    This class represents the signal that will be implanted during a Simulation.
    A signal is represented by a list of motifs, and optionally, positions weights showing where one of the motifs of the signal can
    occur in a sequence.

    A signal is associated with a metadata label, which is assigned to a receptor or repertoire.
    For example antigen-specific/disease-associated (receptor) or diseased (repertoire).


    Arguments:

        motifs (list): A list of the motifs associated with this signal, either defined by seed or by position weight matrix. Alternatively, it can be a list of a list of motifs, in which case the motifs in the same sublist (max 2 motifs) have to co-occur in the same sequence

        sequence_position_weights (dict): a dictionary specifying for each IMGT position in the sequence how likely it is for signal to be there. For positions not specified, the probability of having the signal there is 0.

        v_call (str): V gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;

        j_call (str): J gene with allele if available that has to co-occur with one of the motifs for the signal to exist; can be used in combination with rejection sampling, or full sequence implanting, otherwise ignored; to match in a sequence for rejection sampling, it is checked if this value is contained in the same field of generated sequence;


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        signals:
            my_signal:
                motifs:
                    - my_simple_motif
                    - my_gapped_motif
                sequence_position_weights:
                    109: 0.5
                    110: 0.5
                v_call: TRBV1
                j_call: TRBJ1

    """
    id: str
    motifs: List[Union[Motif, List[Motif]]]
    sequence_position_weights: dict = None
    v_call: str = None
    j_call: str = None

    def get_all_motif_instances(self, sequence_type: SequenceType):
        motif_instances = []
        for motif_group in self.motifs:
            if isinstance(motif_group, list):
                motif_instances.append(MotifInstanceGroup([motif.get_all_possible_instances(sequence_type) for motif in motif_group]))
            else:
                motif_instances.append(motif_group.get_all_possible_instances(sequence_type))
        return motif_instances

    def make_motif_instances(self, count, sequence_type: SequenceType):
        return [motif.instantiate_motif(sequence_type=sequence_type) for motif in random.choices(self.motifs, k=count)]

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return str(vars(self))

    @staticmethod
    def get_documentation():
        initial_doc = str(Signal.__doc__)

        valid_implanting_values = str(
            ReflectionHandler.all_nonabstract_subclass_basic_names(SignalImplantingStrategy, 'Implanting', 'signal_implanting/'))[
                                  1:-1].replace("'", "`")

        docs_mapping = {
            "Valid values for this argument are class names of different signal implanting strategies.":
                f"Valid values are: {valid_implanting_values}"
        }

        doc = update_docs_per_mapping(initial_doc, docs_mapping)
        return doc
