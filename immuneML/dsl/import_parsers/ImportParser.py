from pathlib import Path
from typing import Tuple

from immuneML.IO.dataset_import.DataImport import DataImport
from immuneML.IO.dataset_import.IReceptorImport import IReceptorImport
from immuneML.data_model.dataset.Dataset import Dataset
from immuneML.data_model.receptor.ChainPair import ChainPair
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.Logger import log, print_log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class ImportParser:

    keyword = "datasets"
    valid_keys = ["format", "params"]

    @staticmethod
    def parse(workflow_specification: dict, symbol_table: SymbolTable, path: Path) -> Tuple[SymbolTable, dict]:

        for key in workflow_specification.keys():
            dataset = ImportParser.parse_dataset(key, workflow_specification[key], path)
            symbol_table.add(key, SymbolType.DATASET, dataset)

        return symbol_table, workflow_specification

    @staticmethod
    @log
    def parse_dataset(key: str, dataset_specs: dict, result_path: Path) -> Dataset:
        location = "ImportParser"

        ParameterValidator.assert_keys(list(dataset_specs.keys()), ImportParser.valid_keys, location, f"datasets:{key}", False)

        valid_formats = ReflectionHandler.all_nonabstract_subclass_basic_names(DataImport, "Import", "IO/dataset_import/")
        ParameterValidator.assert_in_valid_list(dataset_specs["format"], valid_formats, location, "format")

        import_cls = ReflectionHandler.get_class_by_name("{}Import".format(dataset_specs["format"]))
        params = ImportParser._prepare_params(dataset_specs, result_path, key)

        if "is_repertoire" in params:
            ParameterValidator.assert_type_and_value(params["is_repertoire"], bool, location, "is_repertoire")

            if params["is_repertoire"]:
                if import_cls != IReceptorImport:
                    assert "metadata_file" in params, f"{location}: Missing parameter: metadata_file under {key}/params/"
                    ParameterValidator.assert_type_and_value(params["metadata_file"], Path, location, "metadata_file")
            else:
                assert "paired" in params, f"{location}: Missing parameter: paired under {key}/params/"
                ParameterValidator.assert_type_and_value(params["paired"], bool, location, "paired")

                if params["paired"]:
                    assert "receptor_chains" in params, f"{location}: Missing parameter: receptor_chains under {key}/params/"
                    ParameterValidator.assert_in_valid_list(params["receptor_chains"], ["_".join(cp.value) for cp in ChainPair], location, "receptor_chains")

        try:
            dataset = import_cls.import_dataset(params, key)
            dataset.name = key
            ImportParser.log_dataset_info(dataset)
        except KeyError as key_error:
            raise KeyError(f"{key_error}\n\nAn error occurred during parsing of dataset {key}. "
                           f"The keyword {key_error.args[0]} was missing. This either means this argument was "
                           f"not defined under definitions/datasets/{key}/params, or this column was missing from "
                           f"an input data file. ")
        except Exception as ex:
            raise Exception(f"{ex}\n\nAn error occurred while parsing the dataset {key}. See the log above for more details.")

        return dataset

    @staticmethod
    def _prepare_params(dataset_specs: dict, result_path: Path, dataset_name: str):
        params = DefaultParamsLoader.load(ImportParser.keyword, dataset_specs["format"])
        if "params" in dataset_specs.keys():
            params = {**params, **dataset_specs["params"]}
        if "result_path" not in params or params["result_path"] is None:
            params["result_path"] = Path(result_path) / "datasets" / dataset_name
        else:
            params["result_path"] = Path(params["result_path"])

        if "path" in params:
            params["path"] = Path(params["path"])
        if "metadata_file" in params:
            params["metadata_file"] = Path(params["metadata_file"])
        dataset_specs["params"] = params
        return params

    @staticmethod
    def log_dataset_info(dataset: Dataset):
        print_log(f"Imported {dataset.__class__.__name__.split('Dataset')[0].lower()} dataset {dataset.name} with "
                  f"{dataset.get_example_count()} examples.", True)
