import torch
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from immuneML.IO.dataset_export.ImmuneMLExporter import ImmuneMLExporter
from immuneML.caching.CacheHandler import CacheHandler
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.environment.SequenceType import SequenceType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.EncoderHelper import EncoderHelper
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler
import pickle
import uuid
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


class Antiberta2Encoder(DatasetEncoder):
    def __init__(self, name: str = None,pickle_file : str = None,embeddings_file : str = None):
        self.batch_size = 2048
        self.model_name = "alchemab/antiberta2"
        self.tokenizer = RoFormerTokenizer.from_pretrained(self.model_name)
        self.name = name
        self.context = None
        self.pickle_file = pickle_file
        self.embeddings_file = embeddings_file

    @staticmethod
    def build_object(dataset, **params):
        return Antiberta2Encoder(**params)

    def _GetSequence(self, dataset):
        list_sequences = []
        for i in dataset.get_data():
            list_sequences.append(i.sequence_aa)
        return list_sequences

    def _split_characters(self, list_sequences):
        split_seq = []
        for x in list_sequences:
            split_seq.append(" ".join(x))
        return split_seq

    def _sequencetokenizer(self, split_seq):
        input_ids = []
        attention_masks = []
        for seq in split_seq:
            tokens = self.tokenizer(seq, truncation=True, padding='max_length', return_tensors="pt",
                                    add_special_tokens=True, max_length=200)
            input_ids.append(tokens['input_ids'])
            attention_masks.append(tokens['attention_mask'])
        return input_ids, attention_masks

    def _datasetcreation(self, input_ids, attention_masks):
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        dataset = TensorDataset(input_ids, attention_masks)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        return data_loader

    def _embeddigstorage(self, data_loader):
        all_embeddings = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = RoFormerTokenizer.from_pretrained(self.model_name)
        model = RoFormerModel.from_pretrained(self.model_name).to(device)
        model.eval()
        with torch.no_grad():
            total_batches = len(data_loader)
            for batch_idx, batch in enumerate(data_loader):
                input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                first_token_batch = embeddings[:, 0, :]
                all_embeddings.append(first_token_batch.cpu())
            all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings.numpy()

    def set_context(self, context: dict):
        self.context = context
        return self

    def _prepare_caching_params(self, dataset, params: EncoderParams, vectors=None, description: str = ""):
        caching_params = (
            ("example_ids", tuple(dataset.get_example_ids())),
            # ("dataset_type", dataset.__class__.__name__),
            # ("labels", tuple(params.label_config.get_labels_by_name())),
            ("encoding", Antiberta2Encoder.__name__),
            # ("learn_model", params.learn_model),
            # ("encoding_params", tuple([(key, getattr(self, key)) for key in vars(self)])),
        )
        return caching_params

    def PickleFileCreation(self, dataset, params: EncoderParams):
        encoded_database= self.embeddings_file
        full_encoded_embeddings = torch.load(encoded_database)
        encoded_dataset = self.context['dataset'].clone()
        labels = params.label_config.get_labels_by_name()
        example_ids_list = []
        for i in self.context['dataset'].get_example_ids():
            example_ids_list.append(i)
        full_encoded_embeddings_np = np.array(full_encoded_embeddings)
        encoded_dataset.encoded_data = EncodedData(
            examples=full_encoded_embeddings_np,
            encoding=Antiberta2Encoder.__name__,
            example_ids=example_ids_list,
            labels= self.context['dataset'].get_metadata(
                labels) if params.encode_labels else None,
            feature_names=None,
            feature_annotations=None
        )
        full_encoded_dataset = encoded_dataset
        full_encoded_dataset.element_generator.buffer_type = None
        unique_id = uuid.uuid4()
        if len(full_encoded_dataset.get_example_ids()) == full_encoded_dataset.encoded_data.examples.shape[0]:
            pickle_file_name = f"full_encoded_dataset_{unique_id}"  #where to save it ? add a random code to identify it
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(full_encoded_dataset, f)
                print("Full encoded dataset has been saved as full_encoded_dataset(something random here )  .")
        else:
            print(
                f"Full encoded dataset has {len(full_encoded_dataset.get_example_ids())} examples and {full_encoded_dataset.encoded_data.examples.shape[0]}")

        print(f"pickle is ready in your directory with name : {pickle_file_name} .")
        sys.exit()
        return pickle_file_name

    def _encode_and_cache(self, dataset,params: EncoderParams):

        if self.pickle_file is not None: #and os.path.exists(self.pickle_file):  change it to work with different paths
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
            input_ids, attention_masks = self._sequencetokenizer(self._split_characters(self._GetSequence(dataset)))
            data_loader = self._datasetcreation(input_ids, attention_masks)
            all_embeddings = self._embeddigstorage(data_loader)
            full_encoded_dataset = self.context['dataset'].clone()
            labels = params.label_config.get_labels_by_name()
            full_encoded_dataset.encoded_data = EncodedData(
                examples=all_embeddings,
                encoding=Antiberta2Encoder.__name__,
                example_ids=self.context['dataset'].get_example_ids(),
                labels=self.context['dataset'].get_metadata(
                    labels) if params.encode_labels else None,
                feature_names=None,
                feature_annotations=None
            )
            print("encoding the full dataset-without using pickle file or torch file.")
        else :
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
                print(f"Sequence index out of range ")

        return relevant_embeddings

    def get_embeddings_by_example_ids(self, full_encoded_dataset, current_sequence_ids):
        relevant_embeddings = []
        for i in current_sequence_ids:
            if i in full_encoded_dataset.get_example_ids():
                index = full_encoded_dataset.encoded_data.example_ids.index(i)
                relevant_embeddings.append(full_encoded_dataset.encoded_data.examples[index])
                print("getting embeddings using example_ids , no pickle")
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
        # Prepare caching parameters with full dataset
        current_dataset = EncoderHelper.get_current_dataset(dataset, self.context)
        caching_params = self._prepare_caching_params(current_dataset, params)
        # Check if the full encoded dataset is in the cache
        if CacheHandler.exists(caching_params):
            full_encoded_dataset = CacheHandler.memo_by_params(
                caching_params,
                lambda: self._encode_and_cache(current_dataset, params)
            )
        else:
            full_encoded_dataset = self._encode_and_cache(current_dataset, params)
            CacheHandler.add(caching_params, full_encoded_dataset)
            # print(f"Full dataset has been encoded again and cached with params {caching_params}.")
        # Update the encoded dataset with the relevant embeddings
        relevant_embeddings = self.make_subset(dataset, full_encoded_dataset)
        encoded_dataset = dataset.clone()
        encoded_dataset.encoded_data = EncodedData(
            examples=relevant_embeddings,
            encoding=Antiberta2Encoder.__name__,
            example_ids=dataset.get_example_ids(),
            labels=dataset.get_metadata(params.label_config.get_labels_by_name()) if params.encode_labels else None,
            feature_names=None,
            feature_annotations=None
        )

        self.context['dataset'].element_generator.buffer_type = None
        return encoded_dataset  # Return the encoded dataset









