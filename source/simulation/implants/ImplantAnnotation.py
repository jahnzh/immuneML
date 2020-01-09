from source.simulation.implants.MotifInstance import MotifInstance


class ImplantAnnotation:

    def __init__(self, signal_id=None, motif_id=None, motif_instance: MotifInstance = None, position=None):
        self.signal_id = signal_id
        self.motif_id = motif_id
        self.motif_instance = motif_instance
        self.position = position
