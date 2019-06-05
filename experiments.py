import os
from datetime import datetime
from itertools import product
from typing import Dict, Set, List, Any, Tuple

import pandas as pd
from joblib import Parallel, delayed

import params
from train import train_kd, ExperimentResults
from utils import log, merge_dicts, pickle_object, unpickle, hash_dict

ParamSet = Dict[str, Any]
ParamGrid = List[ParamSet]
RunnerUUID = str


class ExperimentRunner:
    def __init__(self, experiment_parameters: Dict[str, Set[Any]], n_jobs: int):
        self.experiment_parameters = experiment_parameters
        self.n_jobs = n_jobs
        self.timestamp: str = datetime.now().strftime('%Y%m%d%H%M%S')

        os.makedirs('results/', exist_ok=True)

    @staticmethod
    def _get_param_grid(parameters: Dict[str, Set[Any]]) -> ParamGrid:
        return [dict(zip(parameters.keys(), t)) for t in product(*parameters.values())]

    @staticmethod
    def _file_path_experiment_results(runner_uuid: RunnerUUID) -> str:
        return f'results/{runner_uuid}_experiment_results.pkl'

    def _experiment_result_exists(self, runner_uuid: RunnerUUID) -> bool:
        return os.path.isfile(self._file_path_experiment_results(runner_uuid))

    def _param_run(self, param_set: ParamSet) -> Tuple[ExperimentResults, RunnerUUID]:
        log(f'Running param set: {param_set}')

        uuid = hash_dict(param_set)

        if self._experiment_result_exists(uuid):
            log('Loading experiment results from cache')
            log(uuid)
            experiment_results = unpickle(self._file_path_experiment_results(uuid))
        else:
            log(f'Running uuid {uuid}')
            experiment_results = train_kd(**param_set)
            pickle_object(experiment_results, self._file_path_experiment_results(uuid))

        return experiment_results, uuid

    @staticmethod
    def _get_dict_from_results(results: ExperimentResults) -> Dict:
        _, accs, f1s = results
        return {'val_acc': max(accs), 'val_f1': max(f1s)}

    def run(self):
        param_grid = self._get_param_grid(self.experiment_parameters)
        if self.n_jobs > 1:
            run_output = Parallel(n_jobs=self.n_jobs)(delayed(self._param_run)(param) for param in param_grid)
        else:
            run_output = [self._param_run(param) for param in param_grid]
        results_enriched = [
            merge_dicts(self._get_dict_from_results(result), param_set,
                        {'runner_uuid': runner_uuid},
                        {'experiment_uuid': self.timestamp})
            for (result, runner_uuid), param_set in zip(run_output, param_grid)
        ]
        pd.DataFrame(results_enriched).to_csv(f'results/results_{self.timestamp}.csv', index=False)


if __name__ == '__main__':
    experiment1 = ExperimentRunner(params.experiments, n_jobs=params.n_jobs)
    experiment1.run()
