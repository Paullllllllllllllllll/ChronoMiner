"""The classify commands: shadow-only decision-model answers about the store.

``classify chunks``, ``classify entries`` and ``classify documents`` ask a
classifier's questions (:mod:`chronominer.common.classifiers`) about the
extraction chunks of input items, the entries of finished documents, or a
selection of each item, and record the answers in the ``classifications``
view of the store. Classification never gates an extraction.
:func:`~chronominer.classify.plan.plan_classify` plans without calls or
writes; :func:`~chronominer.classify.run.run_classify` asks and records.
"""

from chronominer.classify.plan import ClassifyPlan, ClassifyTarget, plan_classify
from chronominer.classify.run import ClassifyResult, TargetResult, run_classify

__all__ = [
    "ClassifyPlan",
    "ClassifyResult",
    "ClassifyTarget",
    "TargetResult",
    "plan_classify",
    "run_classify",
]
