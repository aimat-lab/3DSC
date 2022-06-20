from dataclasses import dataclass
from typing import Optional, Tuple, List

import mlflow
import yaml
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MLflowData:
    git_uri: str
    entry_point: str
    experiment_name: Optional[str] = None
    model_version: Optional[str] = None


@dataclass_json
@dataclass
class ModelData:
    name: str
    type: str
    model_parameters: dict
    data_path: str
    data_config: str


class MLflowModel:

    def __init__(self, conf_path: str):
        self.mlflow_data, self.model_data = self._read_config(conf_path)

    @staticmethod
    def _read_config(conf_path: str) -> Tuple[MLflowData, ModelData]:
        with open(conf_path, 'r') as stream:
            loaded_data = yaml.safe_load(stream)

        mlflow_data_dict = loaded_data['mlflow_data']
        model_data_dict = loaded_data['model_data']
        mlflow_data = MLflowData.from_dict(mlflow_data_dict)
        model_data = ModelData.from_dict(model_data_dict)
        return mlflow_data, model_data

    def prepare_run_parameters(self):
        params = self.model_data.model_parameters
        params["data_config"] = self.model_data.data_config
        params["data_path"] = self.model_data.data_path
        return params

    def run_project(self):
        params = self.prepare_run_parameters()
        mlflow.run(uri=self.mlflow_data.git_uri, entry_point=self.mlflow_data.entry_point,
                   parameters=params, experiment_name=self.mlflow_data.experiment_name,
                   version=self.mlflow_data.model_version)
