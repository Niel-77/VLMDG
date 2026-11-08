"""vlmdg - Vision Language Models for Design Grammar.

This package uses a vision language model (VLM) as a *runtime rule generator*
for grammar-driven planar truss optimization.

Sub-packages
------------
``vlmdg.design``
    Labeled-graph representation of a planar truss (``networkx`` backed),
    serialization, and plotting.
``vlmdg.fea``
    Linear 2D truss finite-element solver and structural constraint checks.
``vlmdg.grammar``
    Grammar rules, the mutable rule registry, and the admissibility gate
    (connectivity, geometry, language closure, Maxwell count, equilibrium-matrix
    rank test).
``vlmdg.optimize``
    Shape-annealing search with stall detection / reheat, search diagnostics,
    and the experiment baselines.
``vlmdg.vlm``
    Provider-agnostic VLM interface with an Anthropic implementation and an
    offline deterministic mock.
``vlmdg.rulegen``
    Prompt construction, response parsing, the execution gate with repair loop,
    and the end-to-end rule-generation pipeline used as the ``on_stall`` hook.
``vlmdg.problems``
    Concrete starting designs / design specs (e.g. the cantilever benchmark).
"""

from vlmdg.config import (
    AnnealConfig,
    DesignSpec,
    Material,
    RunConfig,
    VLMConfig,
    load_env,
)

__all__ = [
    "AnnealConfig",
    "DesignSpec",
    "Material",
    "RunConfig",
    "VLMConfig",
    "load_env",
    "__version__",
]

__version__ = "0.1.0"
