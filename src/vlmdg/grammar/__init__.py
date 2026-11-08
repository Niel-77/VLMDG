"""Grammar rules, the mutable rule registry, and the admissibility gate."""

from vlmdg.grammar.base_truss import BASE_TRUSS_RULES, base_truss_grammar
from vlmdg.grammar.registry import Grammar, RuleStats
from vlmdg.grammar.rule import FunctionRule, Rule, RuleApplication
from vlmdg.grammar.validate import GateReport, admissibility_gate, equilibrium_matrix

__all__ = [
    "Rule",
    "FunctionRule",
    "RuleApplication",
    "Grammar",
    "RuleStats",
    "BASE_TRUSS_RULES",
    "base_truss_grammar",
    "GateReport",
    "admissibility_gate",
    "equilibrium_matrix",
]
