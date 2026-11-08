"""VLM-driven rule generation: prompt -> parse -> execution gate -> admissibility gate."""

from vlmdg.rulegen.parser import extract_code_block
from vlmdg.rulegen.pipeline import RuleGenerator, RuleProposal
from vlmdg.rulegen.sandbox import RuleCompileError, compile_rule, run_execution_gate

__all__ = [
    "RuleGenerator",
    "RuleProposal",
    "extract_code_block",
    "compile_rule",
    "run_execution_gate",
    "RuleCompileError",
]
