"""Helpers for v2 ``trainRecipe`` payloads.

``GET .../v2/trainings/recipe`` returns a ready-to-submit ``template``;
callers edit it and submit it via ``rfapi.create_training_v2``. The server
dense-fills omitted defaults server-side.
"""

from __future__ import annotations

import copy
from typing import Any, Dict


def fold_epochs_into_recipe(recipe: Dict[str, Any], epochs: int) -> Dict[str, Any]:
    """Return a copy of *recipe* with *epochs* folded into its hyperparameters.

    The server dense-fills a submitted recipe's hyperparameters (including
    a default ``epochs``) and resolves them ahead of the request body's
    top-level ``epochs``, so a top-level value submitted alongside a recipe
    would otherwise be silently ignored.

    An ``"epochs"`` already set in the recipe's hyperparameters wins; the
    ``hyperparameters`` key is created when a hand-written recipe omits it.
    The input recipe is not mutated.
    """
    folded = copy.deepcopy(recipe)
    hyperparameters = dict(folded.get("hyperparameters") or {})
    hyperparameters.setdefault("epochs", epochs)
    folded["hyperparameters"] = hyperparameters
    return folded
