# --- Semantic Similarity Evaluation ---

import traceback
from utils import truncate_text

def evaluate_semantic_similarity(task, response_text, model, util, threshold=0.7):
    """Evaluates response using sentence embedding cosine similarity."""
    # Args: task (dict): Task def. response_text (str): Model output. model: Loaded SentenceTransformer model. util: SentenceTransformer util module. threshold (float): Min similarity for PASS (0-1).
    # Returns: tuple(float, str): (score_percentage, details_message)
    task_name = task.get("name", "Unnamed Task")

    # --- Validate Task Setup ---
    reference_list = task.get("expected_keywords", [])
    # Semantic requires exactly one reference string in 'expected_keywords'
    if not isinstance(reference_list, list) or len(reference_list) != 1 or not isinstance(reference_list[0], str):
        return 0.0, f"[CONFIG ERROR] Task '{task_name}': Semantic eval requires 'expected_keywords' to be a list with exactly one reference summary string."

    reference_summary = reference_list[0].strip()
    response_clean = response_text.strip()

    # --- Validate Inputs ---
    if not response_clean:
        return 0.0, "Response is empty, cannot calculate semantic similarity."
    if not reference_summary:
        return 0.0, f"[CONFIG ERROR] Task '{task_name}': Reference summary in 'expected_keywords' is empty."

    # Validate and use threshold from task if present
    eval_threshold = task.get("similarity_threshold", threshold)
    if not isinstance(eval_threshold, (float, int)) or not (0 <= eval_threshold <= 1):
         print(f"[WARN] Task '{task_name}': Invalid similarity_threshold '{eval_threshold}'. Using default {threshold:.2f}.")
         eval_threshold = threshold

    # --- Perform Similarity Calculation ---
    try:
        # Ensure inputs are strings
        if not isinstance(reference_summary, str) or not isinstance(response_clean, str):
             return 0.0, "Inputs for semantic similarity must be strings."

        # Generate embeddings
        # Use batch processing if comparing multiple responses to one reference later?
        # For now, one pair at a time.
        embedding_ref = model.encode(reference_summary, convert_to_tensor=True)
        embedding_resp = model.encode(response_clean, convert_to_tensor=True)

        # Compute cosine similarity
        # util.pytorch_cos_sim returns a tensor, get the scalar value
        cosine_score = util.pytorch_cos_sim(embedding_ref, embedding_resp).item()

        # Clamp score to [0, 1] just in case, though should be within [-1, 1]
        # Cosine similarity can be negative, but for text similarity, we usually care about 0..1 range.
        # Let's map [-1, 1] to [0, 100]? Or just take max(0, score)? Let's scale [0,1] to [0,100].
        similarity_score = max(0.0, cosine_score) # Treat negative correlation as 0 similarity for scoring
        score_percent = min(100.0, similarity_score * 100.0) # Scale to 0-100

        detail = (f"Semantic Similarity Score={score_percent:.1f}% "
                  f"(Raw Cosine: {cosine_score:.3f}). "
                  f"Threshold for Pass >= {eval_threshold*100:.1f}%")
        return score_percent, detail

    except Exception as e:
        print(f"[ERROR] Error during semantic similarity calculation for task '{task_name}': {e}")
        traceback.print_exc()
        return 0.0, f"Internal error during semantic similarity calculation: {e}"

