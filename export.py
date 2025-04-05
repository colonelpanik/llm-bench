# --- Results Export Functions ---

import json
import csv
import pathlib
import traceback
from datetime import datetime

from utils import format_na # For consistent NA formatting

def export_summary_csv(results, output_dir, benchmark_name):
    """Exports the summary statistics for each model to a CSV file."""
    # Args: results (dict): Benchmark results. output_dir (pathlib.Path): Directory to save file. benchmark_name (str): Name for filename.
    # Returns: pathlib.Path | None: Path to saved CSV or None on failure.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_bench_name = "".join(c if c.isalnum() else "_" for c in benchmark_name)
    filename = f"summary_{safe_bench_name}_{timestamp}.csv"
    filepath = output_dir / filename

    summaries = []
    for model, data in results.items():
        if model.startswith("_") or not isinstance(data, dict): continue
        summary = data.get("_summary")
        if summary and isinstance(summary, dict):
            summaries.append(summary)
        else:
            # Include skipped/errored models with minimal info
            summaries.append({"model_name": model, "status": data.get("status", "Unknown/Error")})

    if not summaries:
        print("[WARN] No summary data found to export to CSV.")
        return None

    # Define common headers - order matters for CSV
    headers = [
        "model_name", "provider", "status", "overall_weighted_score", "ollama_perf_score",
        "accuracy", "correct_count", "success_count", "processed_count", "api_errors",
        "error_rate", "avg_time_per_task", "total_time", "partial_score_avg",
        "scored_task_count", "tokens_per_sec_avg", "peak_ram_mb", "initial_ram_mb",
        "delta_ram_mb", "peak_gpu_mem_mb", "initial_gpu_mem_mb", "delta_gpu_mem_mb"
    ]

    # Add any headers found in summaries but not in the default list
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())
    extra_headers = sorted([k for k in all_keys if k not in headers and k != 'per_type']) # Exclude complex 'per_type'
    final_headers = headers + extra_headers

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_headers, extrasaction='ignore') # Ignore keys not in headers
            writer.writeheader()
            for summary_data in summaries:
                # Format numeric values for consistency if needed, or write raw
                # Using DictWriter handles missing keys automatically (writes empty string)
                writer.writerow(summary_data)
        print(f"[INFO] Summary exported to CSV: {filepath.resolve()}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to export summary to CSV {filepath}: {e}")
        traceback.print_exc()
        return None


def export_details_json(results, output_dir, benchmark_name):
    """Exports the full detailed results (excluding summaries) to a JSON file."""
    # Args: results (dict): Benchmark results. output_dir (pathlib.Path): Directory to save file. benchmark_name (str): Name for filename.
    # Returns: pathlib.Path | None: Path to saved JSON or None on failure.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_bench_name = "".join(c if c.isalnum() else "_" for c in benchmark_name)
    filename = f"details_{safe_bench_name}_{timestamp}.json"
    filepath = output_dir / filename

    details_data = {}
    for model, data in results.items():
        if model.startswith("_") or not isinstance(data, dict): continue
        # Exclude the summary key, keep all task results
        model_details = {k: v for k, v in data.items() if k != "_summary"}
        if model_details: # Only add if there are task results
             details_data[model] = model_details

    if not details_data:
        print("[WARN] No detailed task data found to export to JSON.")
        return None

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(details_data, jsonfile, indent=2, default=str) # Use default=str for potential non-serializable types
        print(f"[INFO] Detailed results exported to JSON: {filepath.resolve()}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to export details to JSON {filepath}: {e}")
        traceback.print_exc()
        return None