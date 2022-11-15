"""precision metric."""
from sklearn.metrics import precision_score
import datasets
_DESCRIPTION = """\
The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of true positives and ``fp`` the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
"""
_KWARGS_DESCRIPTION = """
    Args:
        predictions: Predicted labels, as returned by a model.
        references: Ground truth labels.
        average: This parameter is required for multiclass/multilabel targets.
        average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, \
            default='binary'
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
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
class Precision(datasets.Metric):
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
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html"],
        )
    def _compute(self, predictions, references, average="macro"):
        return {
            "precision": float(
                precision_score(references, predictions, average=average)
            )
        }