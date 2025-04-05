# --- Main Response Evaluation Dispatcher ---

import traceback
from .standard_evaluators import (
    evaluate_with_regex, partial_keyword_scoring, evaluate_with_confidence,
    simple_keyword_match, simple_keyword_check_strict
)
from .structured_evaluators import evaluate_structured_output
from .code_evaluator import execute_and_test_code # Keep import
from .semantic_evaluator import evaluate_semantic_similarity

def evaluate_response(task, response_text, runtime_config):
    """Evaluates a model's response based on the task definition."""
    # Args: task (dict): The task definition.
    #       response_text (str): The model's response.
    #       runtime_config (RuntimeConfig): Config obj with availability flags and settings.
    # Returns: tuple: (metric, details_string) where metric is bool or float score (0-100).
    task_type = task.get("type", "unknown")
    eval_method = task.get("evaluation_method")
    task_name = task.get("name", "Unnamed Task")

    if not response_text.strip() and task_type != "classification_confidence": # Allow empty for confidence tasks initially
         return False, "No response received from model"

    # --- Specific Evaluation Methods FIRST ---

    # 1. Semantic Similarity
    if eval_method == "semantic" and task_type in ["summarization", "paraphrase", "text_similarity", "semantic_check"]:
        if runtime_config.sentence_transformers_available and runtime_config.semantic_model:
            try:
                # Pass threshold from task config, falling back to default (derived from passing score)
                return evaluate_semantic_similarity(
                    task, response_text, runtime_config.semantic_model, runtime_config.st_util,
                    task.get("similarity_threshold", runtime_config.passing_score_threshold / 100.0)
                )
            except Exception as e:
                print(f"[ERROR] Semantic evaluation failed for task '{task_name}': {e}")
                traceback.print_exc()
                return 0.0, f"Internal error during semantic evaluation: {e}"
        else:
            print(f"[WARN] Task '{task_name}' requested semantic eval, but unavailable/disabled. Falling back...")
            # Fall through


    # 2. Code Execution
    if task_type == "code_generation":
        function_name = task.get("function_name")
        test_cases = task.get("test_cases")
        if function_name and test_cases is not None:
             # Pass code execution timeout from runtime_config
             # *** CHANGE HERE: Pass timeout value ***
             return execute_and_test_code(
                 response_text,
                 function_name,
                 test_cases,
                 runtime_config.code_exec_timeout # Pass the timeout value
             )
        else:
            return False, "[CONFIG ERROR] 'code_generation' task missing 'function_name' or 'test_cases'"


    # 3. Classification with Confidence
    if task_type == "classification_confidence":
        # Pass the default min confidence from runtime_config
        return evaluate_with_confidence(task, response_text, runtime_config.default_min_confidence)


    # 4. Structured Output (JSON/YAML)
    if task_type in ["json_gen", "json_fix", "yaml_gen", "yaml_fix"] or "expected_structure" in task:
        return evaluate_structured_output(task, response_text, runtime_config.pyyaml_available, runtime_config.yaml)


    # 5. Regex-Based Extraction and Validation
    if "expected_regex_map" in task:
        return evaluate_with_regex(task, response_text)


    # --- Keyword-Based Methods (Lower Priority) ---

    # 6. Partial/Weighted Keyword Scoring
    keywords_present_for_scoring = "weighted_keywords" in task or "expected_keywords" in task
    if keywords_present_for_scoring and task_type in ["summarization", "code_explanation", "code_debugging", "paraphrase", "text_similarity", "feature_extraction"]:
        return partial_keyword_scoring(task, response_text)


    # 7. Simple Keyword Matching
    if "expected_keywords" in task:
        expected_keywords = task.get("expected_keywords", [])
        if not expected_keywords:
             pass # Fall through
        elif task_type in ["sentiment", "intent", "reasoning", "classification_simple"]:
            return simple_keyword_match(task, response_text)
        else:
            return simple_keyword_check_strict(task, response_text)


    # --- Final Fallback ---
    return False, f"No specific evaluation criteria in task definition matched for task '{task_name}' (type '{task_type}')"