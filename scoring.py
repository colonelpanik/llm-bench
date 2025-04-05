# --- Performance Score Calculation ---

# Remove direct import of weights from config
# from config import W_ACC_OLLAMA, W_TOKPS_OLLAMA, W_RAM_OLLAMA

def compute_performance_scores(results, current_category_weights, current_default_weight):
    """Computes Ollama Perf Score and Overall Weighted Score."""
    # Args: results (dict): Benchmark results.
    #       current_category_weights (dict): Weights for this run (from config/CLI).
    #       current_default_weight (float): Fallback weight for this run.
    # Returns: None: Modifies the results dictionary in-place.

    summary_data = [] # List of summary dicts for easier processing
    task_categories_map = results.get("_task_categories", {}) # Category -> [task_names]
    task_definitions = results.get("_task_definitions", {}) # task_name -> task_def

    # Get Ollama weights from results metadata if available, else use base defaults
    # This ensures re-scoring uses the weights active *during the run* if possible
    # For new runs, these will match the current RuntimeConfig anyway.
    # TODO: Decide if re-scoring should *always* use current weights instead? For now, use saved weights.
    saved_ollama_weights = results.get("_ollama_score_weights_used", {
        "accuracy": 0.5, "tokens_per_sec": 0.3, "ram_efficiency": 0.2 # Fallback if not saved
    })
    W_ACC = saved_ollama_weights.get('accuracy', 0.5)
    W_TOKPS = saved_ollama_weights.get('tokens_per_sec', 0.3)
    W_RAM = saved_ollama_weights.get('ram_efficiency', 0.2)


    # Reverse map: task_type -> category (approximation if needed)
    type_to_category = {}
    for category, task_names in task_categories_map.items():
        for t_name in task_names:
            if t_name in task_definitions:
                 t_type = task_definitions[t_name].get("type")
                 if t_type and t_type not in type_to_category:
                      type_to_category[t_type] = category

    for model, data in results.items():
        if model.startswith("_"): continue # Skip metadata keys
        summary = data.get("_summary")
        if summary and summary.get("status") == "Completed":
            summary["model"] = summary.get("model_name", model) # Ensure model name is present
            summary["ollama_perf_score"] = None
            summary["overall_weighted_score"] = None
            summary_data.append(summary)
        elif summary:
             print(f"[INFO] Model '{model}' status: {summary.get('status')}. Skipping scoring.")
             if summary:
                 summary["ollama_perf_score"] = None
                 summary["overall_weighted_score"] = None


    # --- 1. Calculate Ollama Performance Score ---
    ollama_summaries = [s for s in summary_data if s.get("provider") == "ollama"]
    ollama_complete = [s for s in ollama_summaries if s.get("accuracy") is not None and
                       s.get("tokens_per_sec_avg") is not None and s["tokens_per_sec_avg"] > 0 and
                       s.get("delta_ram_mb") is not None]

    if ollama_complete:
        print(f"[INFO] Calculating Ollama Performance Score for {len(ollama_complete)} models using weights: Acc={W_ACC}, TokPS={W_TOKPS}, RAM={W_RAM}")
        accuracies = [s["accuracy"] for s in ollama_complete]
        tokps_list = [s["tokens_per_sec_avg"] for s in ollama_complete]
        ram_deltas = [s["delta_ram_mb"] for s in ollama_complete]

        acc_min, acc_max = (min(accuracies), max(accuracies)) if accuracies else (0, 100)
        tokps_min, tokps_max = (min(tokps_list), max(tokps_list)) if tokps_list else (0, 1)
        ram_min, ram_max = (min(ram_deltas), max(ram_deltas)) if ram_deltas else (0, 1)

        if acc_min == acc_max: acc_max += 1e-6
        if tokps_min == tokps_max: tokps_max += 1e-6
        if ram_min == ram_max: ram_max += 1e-6


        def normalize(value, min_val, max_val):
            if max_val == min_val: return 0.5
            value = max(min_val, min(value, max_val))
            return (value - min_val) / (max_val - min_val)

        for s in ollama_complete:
            acc_norm = normalize(s["accuracy"], acc_min, acc_max)
            tokps_norm = normalize(s["tokens_per_sec_avg"], tokps_min, tokps_max)
            ram_delta_norm = normalize(s["delta_ram_mb"], ram_min, ram_max)
            ram_norm_inverted = 1.0 - ram_delta_norm

            score = (W_ACC * acc_norm) + (W_TOKPS * tokps_norm) + (W_RAM * ram_norm_inverted)

            s["ollama_perf_score"] = max(0.0, min(100.0, score * 100.0))

    elif ollama_summaries:
         print("[INFO] No Ollama models had complete metrics (Acc, Tok/s, RAM Delta) for Ollama Performance Score.")


    # --- 2. Calculate Overall Weighted Score (All Models) ---
    # Use the category weights passed in from the current execution context (CLI/Config)
    print(f"[INFO] Calculating Overall Weighted Score using current weights: {current_category_weights} (default: {current_default_weight})")
    for s in summary_data:
        total_weighted_score_points = 0.0
        total_category_weight_sum = 0.0
        per_type_data = s.get("per_type", {})

        if not per_type_data or s.get('processed_count', 0) == 0:
             s["overall_weighted_score"] = 0.0
             continue

        for task_type, type_stats in per_type_data.items():
            if type_stats.get("count", 0) == 0: continue

            category = type_to_category.get(task_type, "default")
            # Use the weights from the *current* config/CLI for scoring
            weight = current_category_weights.get(category, current_default_weight)
            type_acc = type_stats.get("accuracy", 0.0)

            total_weighted_score_points += type_acc * weight
            total_category_weight_sum += weight

        if total_category_weight_sum > 0:
            final_score = total_weighted_score_points / total_category_weight_sum
            s["overall_weighted_score"] = max(0.0, min(100.0, final_score))
        else:
            s["overall_weighted_score"] = 0.0