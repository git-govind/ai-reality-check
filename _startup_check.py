"""
Startup diagnostic — run with: python _startup_check.py
Checks every import that app.py and its pages require.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        errors.append((label, e))

print("=== app.py startup diagnostic ===\n")

# Core utils
check("utils.scoring_utils",  lambda: __import__("utils.scoring_utils"))
check("utils.logging_utils",  lambda: __import__("utils.logging_utils"))
check("utils.cache_utils",    lambda: __import__("utils.cache_utils"))
check("utils.text_utils",     lambda: __import__("utils.text_utils"))
check("utils.image_utils",    lambda: __import__("utils.image_utils"))

# Config / profiler
check("config_loader",        lambda: __import__("config_loader"))
check("profiler",             lambda: __import__("profiler"))

# Text pipeline
check("src.evaluation.factual_checker",      lambda: __import__("src.evaluation.factual_checker"))
check("src.evaluation.consistency_checker",  lambda: __import__("src.evaluation.consistency_checker"))
check("src.evaluation.bias_safety_checker",  lambda: __import__("src.evaluation.bias_safety_checker"))
check("src.evaluation.clarity_scorer",       lambda: __import__("src.evaluation.clarity_scorer"))
check("src.scoring.scoring_engine",          lambda: __import__("src.scoring.scoring_engine"))
check("src.utils.pipeline",                  lambda: __import__("src.utils.pipeline"))

# Image pipeline
check("image_evaluator.metadata_checker",         lambda: __import__("image_evaluator.metadata_checker"))
check("image_evaluator.pixel_forensics",           lambda: __import__("image_evaluator.pixel_forensics"))
check("image_evaluator.ai_artifact_classifier",    lambda: __import__("image_evaluator.ai_artifact_classifier"))
check("image_evaluator.image_text_consistency",    lambda: __import__("image_evaluator.image_text_consistency"))
check("image_evaluator.image_scoring",             lambda: __import__("image_evaluator.image_scoring"))
check("image_evaluator.evaluate_image",            lambda: __import__("image_evaluator.evaluate_image"))

# Models
check("models.llm_registry",        lambda: __import__("models.llm_registry"))
check("models.embeddings_registry",  lambda: __import__("models.embeddings_registry"))

print(f"\n=== {len(errors)} error(s) ===")
for label, e in errors:
    print(f"  {label}: {type(e).__name__}: {e}")
