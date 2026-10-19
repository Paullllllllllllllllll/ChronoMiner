"""Batch extraction on the shared backends of OpenAI, Anthropic and Google.

:mod:`~chronominer.batch.bodies` builds the request bodies,
:mod:`~chronominer.batch.submit` submits a document's chunks as store
submissions, :mod:`~chronominer.batch.finalize` finalizes a finished batch
into the store for batch check (:mod:`~chronominer.batch.check`) and repair
(:mod:`~chronominer.batch.repair`), and :mod:`~chronominer.batch.cancel`
cancels and lists batches; the shared
:mod:`~chronominer.common.batch_reconcile` settles unresolved submissions.
Every API call goes through :mod:`~chronominer.batch.clients`.
"""
