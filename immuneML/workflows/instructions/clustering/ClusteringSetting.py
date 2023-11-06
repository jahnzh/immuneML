from dataclasses import dataclass

from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.ml_methods.clustering.ClusteringMethod import ClusteringMethod
from immuneML.ml_methods.dim_reduction.DimRedMethod import DimRedMethod


@dataclass
class ClusteringSetting:
    encoder: DatasetEncoder
    encoder_params: dict
    encoder_name: str
    clustering_method: ClusteringMethod
    clustering_params: dict
    clustering_method_name: str
    dim_reduction_method: DimRedMethod = None
    dim_red_params: dict = None
    dim_red_name: str = None

    def get_key(self) -> str:
        key = self.encoder_name
        if self.dim_red_name:
            key += f"_{self.dim_red_name}"
        key += f"_{self.clustering_method_name}"
        return key

    def __str__(self):
        return self.get_key()
