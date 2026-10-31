import os

# PyTorch ships its own libomp. Our engine's libomp must be initialised first, otherwise the
# second runtime crashes on its first parallel region; so load the engines before any torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .engine import omp_engine, serial_engine  # noqa: E402

omp_engine()
serial_engine()
