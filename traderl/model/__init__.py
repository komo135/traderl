from traderl.model.cnnnet import cnnnet_models
from traderl.model.build import build_model

__all__ = ["build_model", "all_models"]

all_models = {}

all_models.update(cnnnet_models)
