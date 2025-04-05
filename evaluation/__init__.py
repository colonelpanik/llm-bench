# Evaluation Package Initializer

# Make the main evaluator function easily accessible
from .evaluator import evaluate_response

# You could potentially import other specific evaluators if needed directly
# from .standard_evaluators import partial_keyword_scoring
# from .code_evaluator import execute_and_test_code
# etc.

__all__ = ['evaluate_response']
