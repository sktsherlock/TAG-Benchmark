"""recall metric."""
from sklearn.metrics import recall_score
import datasets
_DESCRIPTION = """\
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and 
    fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
"""
_KWARGS_DESCRIPTION = """
    Args:
        predictions: Predicted labels, as returned by a model.
        references: Ground truth labels.
        average: This parameter is required for multiclass/multilabel targets.
"""
_CITATION = """\
    @article{scikit-learn,
        title={Scikit-learn: Machine Learning in {P}ython}, 
        author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
                and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
                and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and 
                Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},   
        journal={Journal of Machine Learning Research},
        volume={12},
        pages={2825--2830},
        year={2011}
    }
"""
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Recall(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html"],
        )
    def _compute(self, predictions, references, average="macro"):
        return {
            "recall": float(
                recall_score(references, predictions, average=average)
            )
        }