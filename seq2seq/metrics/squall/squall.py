"""squall metrics."""

from typing import Optional, Union
from seq2seq.metrics.spider.spider_test_suite import compute_test_suite_metric
from seq2seq.metrics.spider.spider_exact_match import compute_exact_match_metric
import datasets

from data.squall.model.evaluator import Evaluator
from typing import Dict, Any

_DESCRIPTION = """
Squall metrics.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """\
@inproceedings{Shi:Zhao:Boyd-Graber:Daume-III:Lee-2020,
	Title = {On the Potential of Lexico-logical Alignments for Semantic Parsing to {SQL} Queries},
	Author = {Tianze Shi and Chen Zhao and Jordan Boyd-Graber and Hal {Daum\'{e} III} and Lillian Lee},
	Booktitle = {Findings of EMNLP},
	Year = {2020},
}
"""

_URL = "https://raw.githubusercontent.com/tzshi/squall/main/data/"
_URLS = {
    "squall": _URL + "squall.json",
    "wtq-test": _URL + "wtq-test.json",
    "dev-0": _URL +  "dev-0.ids",
    "dev-1": _URL +  "dev-1.ids",
    "dev-2": _URL +  "dev-2.ids",
    "dev-3": _URL +  "dev-3.ids",
    "dev-4": _URL +  "dev-4.ids",
}

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Squall(datasets.Metric):
    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )
        self.test_suite_db_dir: Optional[str] = kwargs.pop("test_suite_db_dir", None)

    def _info(self):
        if self.config_name not in [
            "execution_accuracy",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in " '["execution_accuracy"]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": {
                        "query": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "nt": datasets.Value("string"),
                        "header": datasets.features.Sequence(datasets.Value("string")),
                        "context": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                        "db_path": datasets.Value("string"),
                        "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                        "db_column_names": datasets.features.Sequence(
                            {
                                "table_id": datasets.Value("int32"),
                                "column_name": datasets.Value("string"),
                            }
                        ),
                        "db_foreign_keys": datasets.features.Sequence(
                            {
                                "column_id": datasets.Value("int32"),
                                "other_column_id": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            reference_urls=[_URL],
        )

    def _postprocess(self, predictions, references):

        # preds and labels for all eval samples
        # prepare the prediction format for the wtq evaluator
        predictions = []
        for pred, ref in enumerate(zip(predictions, references)):
            table_id = ref['db_id']
            nt_id = ref['nt']
            header = ref['header']
            # repalce the natural language header with c1, c2, ... headers
            for j, h in enumerate(header):
                pred=pred.replace(h, 'c'+str(j+1))
            result_dict = {"sql": pred, "id": nt_id, "tgt": ref['query']}
            res = {"table_id": table_id, "result": [result_dict]}
            predictions.append(res)

        return predictions

    
    def _compute_execuntion_accuracy(self, predictions, references) -> Dict[str, Any]:
        total = len(predictions)
        evaluator = Evaluator(
                f"/workspaces/picard/data/squall/tables/tagged/",
                f"/workspaces/picard/data/squall/tables/db/",
                f"/workspaces/picard/Third_party/stanford-corenlp-full-2018-10-05/"
        )
        predictions = self._postprocess(predictions, references)
        ex_accu = evaluator.evaluate(predictions)
        lf_accu = 0
        for d in predictions:
            if d['result'][0]['sql'] == d['result'][0]['tgt']:
                lf_accu += 1

        return {
                "execution_accuracy": ex_accu/total, 
                "logical_form_accuracy": lf_accu/total
        }


    def _compute(self, predictions, references):

        if self.config_name == "execution_accuracy":
            res = self._compute_execuntion_accuracy(predictions, references)
        else:
            res = dict()

        return {**res}
