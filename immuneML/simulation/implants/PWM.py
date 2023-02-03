from pathlib import Path

from bionumpy.io.motifs import read_motif
from bionumpy.sequence.position_weight_matrix import PWM as bnp_PWM

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType


class PWM:
    """
    Class describing positional weight matrix and using bionumpy's PWM internally.

    Arguments:
        file_path: path to the file where the PWM is stored

        threshold (float): when matching PWM to a sequence, this is the threshold to consider the sequence as containing the motif

    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            my_custom_pwm: # this will be the identifier of the motif
                file_path: my_pwm_1.csv
                threshold: 2

    """

    def __init__(self, file_path: Path, matrix: bnp_PWM, threshold: float):

        self.file_path = file_path
        self.matrix = matrix
        self.threshold = threshold

        assert self.matrix is not None or self.file_path is not None, (file_path, matrix)

    @classmethod
    def build(cls, file_path, threshold: float):
        assert Path(file_path).is_file(), file_path
        matrix = read_motif(file_path)
        return PWM(file_path, matrix, threshold)

    def get_all_motif_instances(self, sequence_type: SequenceType):
        assert sorted(self.matrix.alphabet) == sorted(EnvironmentSettings.get_sequence_alphabet(sequence_type))
        return self
