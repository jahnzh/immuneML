# import torch
# import sys
# from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
# import argparse
# import numpy as np
# import pickle
# import torch
# from transformers import RoFormerTokenizer, RoFormerModel
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import argparse
# from collections import Counter
# from sklearn.feature_extraction import DictVectorizer
# from immuneML.data_model.encoded_data.EncodedData import EncodedData
# from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
# from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
# from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
# from immuneML.data_model.encoded_data.EncodedData import EncodedData
# from immuneML.encodings.DatasetEncoder import DatasetEncoder
# from immuneML.encodings.DatasetEncoder import DatasetEncoder
# from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
# from immuneML.encodings.EncoderParams import EncoderParams
# from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
# from immuneML.util.ParameterValidator import ParameterValidator
# from immuneML.util.PathBuilder import PathBuilder
# from immuneML.environment.SequenceType import SequenceType
# from immuneML.util.EncoderHelper import EncoderHelper
# from immuneML.caching.CacheHandler import CacheHandler
# import tempfile
# import os
# from tqdm import tqdm
# 
# 
# class ESM2Encoder(DatasetEncoder):
#     def __init__(self, name: str = None):
#         self.MODEL_LOCATION = "esm2_t33_650M_UR50D"
#         self.TOKS_PER_BATCH = 4096
#         self.REPR_LAYERS = [-1]
#         # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model, self.alphabet = pretrained.load_model_and_alphabet(self.MODEL_LOCATION)
#         self.name = name
#         self.context = None
# 
#     @staticmethod
#     def build_object(dataset, **params):
#         return ESM2Encoder(**params)
# 
#     def _GetSequence(self, dataset):
#         list_sequences = []
#         for i in dataset.get_data():
#             list_sequences.append(i.sequence_aa)
#         return list_sequences
# 
#     def set_context(self, context: dict):
#         self.context = context
#         return self
# 
#     def createFasta(self, list_sequences):
#         fasta_file = tempfile.NamedTemporaryFile(mode="w",
#                                                  delete=False)  # create a fasta file that when is closed is also deleted
#         for i, seq in enumerate(list_sequences):
#             fasta_file.write(f">{i}\n{seq}\n")
#         fasta_file.close()  
#         if os.path.exists(fasta_file.name):
#             print(f"The file {fasta_file.name} has been created.")
#         else:
#             print(f"The file {fasta_file.name} was not created.")
# 
#         return fasta_file
# 
#     def _datasetcreation(self, fasta_file):
# 
#         dataset = FastaBatchedDataset.from_file(fasta_file.name)
#         batches = dataset.get_batch_indices(self.TOKS_PER_BATCH, extra_toks_per_seq=1)
#         data_loader = torch.utils.data.DataLoader(
#             dataset, collate_fn=self.alphabet.get_batch_converter(), batch_sampler=batches
#         )
#         os.remove(fasta_file.name)
#         if os.path.exists(fasta_file.name):
#             print(f"The file {fasta_file.name} is still here.")
#         else:
#             print(f"The file {fasta_file.name} has been used as input and then deleted.")
# 
#         return data_loader, batches
# 
#     def get_embeddings(self, data_loader, batches):
#         self.model.eval()
#         if torch.cuda.is_available():
#             model = self.model.cuda()
# 
#         assert all(-(self.model.num_layers + 1) <= i <= self.model.num_layers for i in self.REPR_LAYERS)
#         repr_layers = [(i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in self.REPR_LAYERS]
#         mean_representations = []
#         seq_labels = []
# 
#         with torch.no_grad():
#             for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader, desc="Processing batches")):
#                 print(
#                     f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
#                 )
#                 if torch.cuda.is_available():
#                     toks = toks.to(device="cuda", non_blocking=True)
# 
#                 out = self.model(toks, repr_layers=repr_layers, return_contacts=False)
# 
#                 representations = {
#                     layer: t.to(device="cpu") for layer, t in out["representations"].items()
#                 }
# 
#                 for i, label in enumerate(labels):
#                     seq_labels.append(label)
#                     mean_representation = [t[i, 1: len(strs[i]) + 1].mean(0).clone()
#                                            for layer, t in representations.items()]
#                     mean_representations.append(mean_representation[0])
# 
#         mean_representations = torch.vstack(mean_representations)
#         ordering = np.argsort([int(i) for i in seq_labels])
#         mean_representations = mean_representations[ordering, :]
#         mean_representations_np = mean_representations.numpy()
#         print(mean_representations.shape)
#         return mean_representations_np
# 
#     def _prepare_caching_params(self, dataset, params: EncoderParams, vectors=None, description: str = ""):
#         caching_params = (
#             ("example_ids", tuple(dataset.get_example_ids())),
#             ("dataset_type", dataset.__class__.__name__),
#             ("labels", tuple(params.label_config.get_labels_by_name())),
#             ("encoding", ESM2Encoder.__name__),
#             #("learn_model", params.learn_model),
#             #("encoding_params", tuple([(key, getattr(self, key)) for key in vars(self)])),
#         )
#         return caching_params
# 
#     def _encode_and_cache(self, dataset, params: EncoderParams):
#         list_sequences = self._GetSequence(dataset)
#         fasta_file = self.createFasta(list_sequences)
#         data_loader, batches = self._datasetcreation(fasta_file)
#         all_embeddings = self.get_embeddings(data_loader, batches)
#         full_encoded_dataset = self.context['dataset'].clone()
#         labels = params.label_config.get_labels_by_name()
#         full_encoded_dataset.encoded_data = EncodedData(
#             examples=all_embeddings,
#             encoding=ESM2Encoder.__name__,
#             example_ids=self.context['dataset'].get_example_ids(),
#             labels=self.context['dataset'].get_metadata(
#                 labels) if params.encode_labels else None,
#             feature_names=None,
#             feature_annotations=None
#         )
#         print("encoding the full dataset")
#         return full_encoded_dataset
# 
#     def get_embeddings_by_example_ids(self, full_encoded_dataset, current_sequence_ids):
#         relevant_embeddings = []
#         for i in current_sequence_ids:
#             if i in full_encoded_dataset.get_example_ids():
#                 index = full_encoded_dataset.encoded_data.example_ids.index(i)
#                 relevant_embeddings.append(full_encoded_dataset.encoded_data.examples[index])
#                 print(f"{relevant_embeddings} was correctly retrived with {index} ")
#             else:
#                 print("Sequence not found in full encoded dataset.")
#         return relevant_embeddings
# 
#     def make_subset(self, dataset, full_encoded_dataset):
#         current_sequence_ids = dataset.get_example_ids()
#         relevant_embeddings_list = self.get_embeddings_by_example_ids(full_encoded_dataset, current_sequence_ids)
#         relevant_embeddings = np.array(relevant_embeddings_list)
#         return relevant_embeddings
# 
#     def encode(self, dataset, params: EncoderParams):
#         current_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
#         caching_params = self._prepare_caching_params(current_dataset, params)
#         # Check if the full encoded dataset is in the cache
#         if CacheHandler.exists(caching_params):
#             #print(f"Retrieving full encoded dataset from cache with params.")
#             full_encoded_dataset = CacheHandler.memo_by_params(
#                 caching_params,
#                 lambda: self._encode_and_cache(current_dataset, params)
#             )
#         else:
#             full_encoded_dataset = self._encode_and_cache(current_dataset, params)
#             CacheHandler.add(caching_params, full_encoded_dataset)
#             print(f"Full dataset has been encoded again and cached with params {caching_params}.")
#         # Update the encoded dataset with the relevant embeddings
#         relevant_embeddings = self.make_subset(dataset, full_encoded_dataset)
# 
#         encoded_dataset = dataset.clone()
#         encoded_dataset.encoded_data = EncodedData(
#             examples=relevant_embeddings,
#             encoding=ESM2Encoder.__name__,
#             example_ids=dataset.get_example_ids(),
#             labels=dataset.get_metadata(params.label_config.get_labels_by_name()) if params.encode_labels else None,
#             feature_names=None,
#             feature_annotations=None
#         )
#         self.context['dataset'].element_generator.buffer_type = None
#         return encoded_dataset

import torch
import sys
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.preprocessing.FeatureScaler import FeatureScaler
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.PathBuilder import PathBuilder
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.caching.CacheHandler import CacheHandler
import tempfile
import os
import uuid
from tqdm import tqdm


class ESM2Encoder(DatasetEncoder):
    def __init__(self, name: str = None, pickle_file: str = None, embeddings_file: str = None, ):
        self.MODEL_LOCATION = "esm2_t33_650M_UR50D"
        self.TOKS_PER_BATCH = 4096
        self.REPR_LAYERS = [-1]
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = pretrained.load_model_and_alphabet(self.MODEL_LOCATION)
        self.name = name
        self.context = None
        self.pickle_file = pickle_file
        self.embeddings_file = embeddings_file

    @staticmethod
    def build_object(dataset, **params):
        return ESM2Encoder(**params)

    def _GetSequence(self, dataset):
        list_sequences = []
        for i in dataset.get_data():
            list_sequences.append(i.sequence_aa)
        return list_sequences

    def set_context(self, context: dict):
        self.context = context
        return self

    def createFasta(self, list_sequences):
        fasta_file = tempfile.NamedTemporaryFile(mode="w",
                                                 delete=False)
        for i, seq in enumerate(list_sequences):
            fasta_file.write(f">{i}\n{seq}\n")
        fasta_file.close()
        if os.path.exists(fasta_file.name):
            print(f"The file {fasta_file.name} has been created.")
        else:
            print(f"The file {fasta_file.name} was not created.")

        return fasta_file

    def _datasetcreation(self, fasta_file):

        dataset = FastaBatchedDataset.from_file(fasta_file.name)
        batches = dataset.get_batch_indices(self.TOKS_PER_BATCH, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=self.alphabet.get_batch_converter(), batch_sampler=batches
        )
        os.remove(fasta_file.name)
        if os.path.exists(fasta_file.name):
            print(f"The file {fasta_file.name} is still here.")
        else:
            print(f"The file {fasta_file.name} has been used as input and then deleted.")

        return data_loader, batches

    def get_embeddings(self, data_loader, batches):
        self.model.eval()
        if torch.cuda.is_available():
            model = self.model.cuda()

        assert all(-(self.model.num_layers + 1) <= i <= self.model.num_layers for i in self.REPR_LAYERS)
        repr_layers = [(i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in self.REPR_LAYERS]
        mean_representations = []
        seq_labels = []

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader, desc="Processing batches")):
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)

                out = self.model(toks, repr_layers=repr_layers, return_contacts=False)

                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }

                for i, label in enumerate(labels):
                    seq_labels.append(label)
                    mean_representation = [t[i, 1: len(strs[i]) + 1].mean(0).clone()
                                           for layer, t in representations.items()]
                    mean_representations.append(mean_representation[0])

        mean_representations = torch.vstack(mean_representations)
        ordering = np.argsort([int(i) for i in seq_labels])
        mean_representations = mean_representations[ordering, :]
        mean_representations_np = mean_representations.numpy()
        print(mean_representations.shape)
        return mean_representations_np

    def _prepare_caching_params(self, dataset, params: EncoderParams, vectors=None, description: str = ""):
        caching_params = (
            ("example_ids", tuple(dataset.get_example_ids())),
            ("dataset_type", dataset.__class__.__name__),
            ("labels", tuple(params.label_config.get_labels_by_name())),
            ("encoding", ESM2Encoder.__name__),
            #("learn_model", params.learn_model),
            #("encoding_params", tuple([(key, getattr(self, key)) for key in vars(self)])),
        )
        return caching_params

    def PickleFileCreation(self, dataset, params: EncoderParams):
        encoded_database = self.embeddings_file
        full_encoded_embeddings_tot = torch.load(encoded_database)
        #full_encoded_embeddings_tot = full_encoded_embeddings_tot[:10]
        first_sequence = full_encoded_embeddings_tot[0]
        print(first_sequence)
        encoded_dataset = self.context['dataset'].clone()
        labels = params.label_config.get_labels_by_name()
        example_ids_list = []
        for i in self.context['dataset'].get_example_ids():
            example_ids_list.append(i)
        full_encoded_embeddings_np = np.array(full_encoded_embeddings_tot)
        encoded_dataset.encoded_data = EncodedData(
            examples=full_encoded_embeddings_np,
            encoding=ESM2Encoder.__name__,
            example_ids=example_ids_list,
            labels=self.context['dataset'].get_metadata(
                labels) if params.encode_labels else None,
            feature_names=None,
            feature_annotations=None
        )
        full_encoded_dataset = encoded_dataset
        full_encoded_dataset.element_generator.buffer_type = None
        unique_id = uuid.uuid4()
        if len(full_encoded_dataset.get_example_ids()) == full_encoded_dataset.encoded_data.examples.shape[0]:
            pickle_file_name = f"full_encoded_dataset_{unique_id}.pkl"  #where to save it ? add a random code to identify it
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(full_encoded_dataset, f)
                print("Full encoded dataset has been saved as full_encoded_dataset(something random here )  .")
        else:
            print(
                f"Full encoded dataset has {len(full_encoded_dataset.get_example_ids())} examples and {full_encoded_dataset.encoded_data.examples.shape[0]}")

        print(f"pickle is ready in your directory with name : {pickle_file_name} .")
        sys.exit()
        return pickle_file_name

    def _encode_and_cache(self, dataset, params: EncoderParams):
        if self.pickle_file is not None:  #change it to work with different paths
            with open(self.pickle_file, "rb") as f:
                encoded_dataset = pickle.load(f)
                full_encoded_dataset = encoded_dataset
            print("encoding the full dataset using input pickle file .")
        elif self.embeddings_file is not None and os.path.exists(self.embeddings_file) and self.pickle_file is None:
            pickle_file_name = self.PickleFileCreation(dataset, params)
            with open(pickle_file_name, "rb") as f:
                encoded_dataset = pickle.load(f)
                full_encoded_dataset = encoded_dataset
                print("encoding the full dataset-using embedding file .")
        elif self.embeddings_file is None and self.pickle_file is None:
            list_sequences = self._GetSequence(dataset)
            fasta_file = self.createFasta(list_sequences)
            data_loader, batches = self._datasetcreation(fasta_file)
            all_embeddings = self.get_embeddings(data_loader, batches)
            full_encoded_dataset = self.context['dataset'].clone()
            labels = params.label_config.get_labels_by_name()
            full_encoded_dataset.encoded_data = EncodedData(
                examples=all_embeddings,
                encoding=ESM2Encoder.__name__,
                example_ids=self.context['dataset'].get_example_ids(),
                labels=self.context['dataset'].get_metadata(
                    labels) if params.encode_labels else None,
                feature_names=None,
                feature_annotations=None
            )
            print("encoding the full dataset")
            print("encoding the full dataset-without using pickle file or torch file.")
        else:
            print("No file found to encode the dataset.")

        return full_encoded_dataset

    def get_embeddings_by_index_ids(self, full_encoded_dataset, current_index_ids):
        relevant_embeddings = []
        for b in current_index_ids:
            example_ids_list = full_encoded_dataset.get_example_ids()
            if b < len(full_encoded_dataset.encoded_data.examples):
                relevant_embeddings.append(full_encoded_dataset.encoded_data.examples[b])
                #print("getting embeddings from index(using pickle)")
                #f" index {b} was added to the relevant embeddings as {full_encoded_dataset.encoded_data.examples[b]} .")
            else:
                print(f"Sequence index out of range {b}")


        return relevant_embeddings

    def get_embeddings_by_example_ids(self, full_encoded_dataset, current_sequence_ids):
        relevant_embeddings = []
        for i in current_sequence_ids:
            if i in full_encoded_dataset.get_example_ids():
                index = full_encoded_dataset.encoded_data.example_ids.index(i)
                relevant_embeddings.append(full_encoded_dataset.encoded_data.examples[index])
                #print(f"{relevant_embeddings} was correctly retrived with {index} ")
            else:
                print("Sequence not found in full encoded dataset.")
        return relevant_embeddings

    def make_subset(self, dataset, full_encoded_dataset):
        if self.pickle_file is not None and os.path.exists(self.pickle_file):
            current_index_ids = []
            for i in dataset.get_data():
                try:
                    current_index_ids.append(int(i.metadata.cell_id))
                except ValueError:
                    print(f"ValueError: {i.metadata.cell_id} is not a valid index.")
            relevant_embeddings_list = self.get_embeddings_by_index_ids(full_encoded_dataset, current_index_ids)
            relevant_embeddings = np.array(relevant_embeddings_list)
            print(f"making subsets with {len(relevant_embeddings)} embeddings.")
        elif self.embeddings_file is None and self.pickle_file is None:
            current_sequence_ids = dataset.get_example_ids()
            relevant_embeddings_list = self.get_embeddings_by_example_ids(full_encoded_dataset, current_sequence_ids)
            relevant_embeddings = np.array(relevant_embeddings_list)

        return relevant_embeddings

    def encode(self, dataset, params: EncoderParams):
        current_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
        caching_params = self._prepare_caching_params(current_dataset, params)
        # Check if the full encoded dataset is in the cache
        if CacheHandler.exists(caching_params):
            #print(f"Retrieving full encoded dataset from cache with params.")
            full_encoded_dataset = CacheHandler.memo_by_params(
                caching_params,
                lambda: self._encode_and_cache(current_dataset, params)
            )
        else:
            full_encoded_dataset = self._encode_and_cache(current_dataset, params)
            CacheHandler.add(caching_params, full_encoded_dataset)
            #print(f"Full dataset has been encoded again and cached with params {caching_params}.")
        # Update the encoded dataset with the relevant embeddings
        relevant_embeddings = self.make_subset(dataset, full_encoded_dataset)

        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=relevant_embeddings,
            encoding=ESM2Encoder.__name__,
            example_ids=dataset.get_example_ids(),
            labels=dataset.get_metadata(params.label_config.get_labels_by_name()) if params.encode_labels else None,
            feature_names=None,
            feature_annotations=None
        )
        self.context['dataset'].element_generator.buffer_type = None
        return encoded_dataset

