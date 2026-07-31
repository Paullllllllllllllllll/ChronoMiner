"""Model capability registry and detection.

* :mod:`modules.config.capabilities.registry` — data (Capabilities dataclass,
  base dicts per provider, static model registry).
* :mod:`modules.config.capabilities.detection` — lookup logic
  (``detect_capabilities``, ``detect_provider``, ``_build_caps``).
"""

from modules.config.capabilities.detection import (
    detect_capabilities,
    detect_provider,
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
]
