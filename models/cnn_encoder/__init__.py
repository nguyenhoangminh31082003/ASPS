from .build_mscan import (
    build_mscan_base,
    build_mscan_large,
    build_mscan_small,
    build_mscan_tiny,
)

mscan_model_registry = {
    "tiny": build_mscan_tiny,
    "small": build_mscan_small,
    "base": build_mscan_base,
    "large": build_mscan_large,
}
