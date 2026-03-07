"""
Evaluation pipeline helper.
Runs all checkers sequentially and returns a consolidated EvaluationReport.
"""
from __future__ import annotations

from src.evaluation import factual_checker, consistency_checker, bias_safety_checker, clarity_scorer
from src.scoring.scoring_engine import EvaluationReport, aggregate


def evaluate(
    prompt: str,
    ai_response: str,
    model_display_name: str | None = None,
    run_llm_critique: bool = True,
) -> EvaluationReport:
    """
    Full evaluation pipeline.

    Args:
        prompt: The original user prompt.
        ai_response: The AI model's response text.
        model_display_name: Model to use for LLM-based critique (optional).
        run_llm_critique: If False, only rule-based checks are run (faster).

    Returns:
        An EvaluationReport with all scores and findings.
    """
    effective_model = model_display_name if run_llm_critique else None

    factual = factual_checker.run(ai_response)
    consistency = consistency_checker.run(prompt, ai_response, effective_model)
    bias = bias_safety_checker.run(ai_response, effective_model)
    clarity = clarity_scorer.run(prompt, ai_response)

    return aggregate(factual, consistency, bias, clarity)
