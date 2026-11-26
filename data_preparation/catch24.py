"""Versioned Catch24 calls shared by the EEG and comparator pipelines."""

from importlib.metadata import version

import numpy as np
import pycatch22


PYCATCH22_VERSION = version("pycatch22")
CATCH24_OPTIONS = {"catch24": True}
if PYCATCH22_VERSION == "1.0.2":
    CATCH24_OPTIONS["mask_nans"] = False


def extract_catch24(signal):
    return pycatch22.catch22_all(signal, **CATCH24_OPTIONS)


def descriptor_names():
    if PYCATCH22_VERSION == "1.0.2":
        return tuple(feature["name"] for feature in pycatch22.feature_table(catch24=True))
    return tuple(extract_catch24(np.arange(128, dtype=float))["names"])
