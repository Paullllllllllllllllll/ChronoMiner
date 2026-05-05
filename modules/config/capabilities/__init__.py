"""Model capability registry, detection, and parameter gating.

Split from the former ``modules/llm/model_capabilities.py`` into three
cohesive files:

* :mod:`modules.config.capabilities.registry` — data (Capabilities dataclass,
  base dicts per provider, static model registry).
* :mod:`modules.config.capabilities.detection` — lookup logic
  (``detect_capabilities``, ``detect_provider``, ``_build_caps``).
* :mod:`modules.config.capabilities.params` — LangChain ``disabled_params``
  computation from a model name or Capabilities instance.
"""

from modules.config.capabilities.detection import (
    detect_capabilities,
    detect_provider,
)
from modules.config.capabilities.params import (
    disabled_params_for_capabilities,
    disabled_params_for_model,
)
from modules.config.capabilities.registry import (
    ApiPref,
    Capabilities,
    ImageDetail,
    ProviderType,
)

__all__ = [
    "Capabilities",
    "ApiPref",
    "ImageDetail",
    "ProviderType",
    "detect_capabilities",
    "detect_provider",
    "disabled_params_for_capabilities",
    "disabled_params_for_model",
]
