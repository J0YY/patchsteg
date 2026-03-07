# Backward-compatibility shim. Use core.defenses.sanitize directly.
from core.defenses.sanitize import *
from core.defenses.sanitize import (
    VaeRoundTripSanitizer, NoisyRoundTripSanitizer,
    LatentQuantizationSanitizer, LatentSmoothingSanitizer,
)
