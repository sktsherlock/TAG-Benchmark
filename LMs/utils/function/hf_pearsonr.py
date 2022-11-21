"""Pearsonr metric."""
from scipy.stats import pearsonr
import datasets

_DESCRIPTION = """\
    The Pearson correlation coefficient measures the linear relationship between two datasets.
    Strictly speaking, Pearson's correlation requires that each dataset be normally distributed.
    Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
"""
_KWARGS_DESCRIPTION = """
    Args:
        predictions: Predicted labels, as returned by a model.
        references: Ground truth labels.
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
class Pearsonr(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("float32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pearsonr.html"],
        )

    def _compute(self, predictions, references):
        return {
            "pearsonr": float(
                pearsonr(references, predictions)[0]
            )
        }
