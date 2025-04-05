# --- HTML Report Generation and Plotting ---

import json
import re
import traceback
from datetime import datetime
from collections import defaultdict
import webbrowser
import pathlib
import csv # Added for CSV export

from utils import format_na, truncate_text
# Conditional import handled in CLI, check flag here
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker

# --- Plotting ---

STAGES = ["None", "Gates", "Persona"]
STAGE_COLORS = {"None": "skyblue", "Gates": "lightcoral", "Persona": "lightgreen"}

def get_model_stage(model_name):
    """Extracts the stage (None, Gates, Persona) from the model name."""
    # Simple check based on keywords in the model name string
    if "Gates" in model_name:
        return "Gates"
    elif "Persona" in model_name:
        return "Persona"
    else:
        return "None"

def generate_ranking_plot(results, metric_key, title, ylabel, output_filename, runtime_config, lower_is_better=False, value_format="{:.1f}", is_percentage=False, group_by_stage=False):
    """Generates a horizontal bar chart ranking models by a metric."""
    # Args: ... (same as before) ... group_by_stage (bool): Color-code bars by stage (None, Gates, Persona).
    # Returns: str | None: Relative path to the saved image ("images/filename.png") or None on failure/skip.

    if not runtime_config.visualizations_enabled or not runtime_config.matplotlib_available:
        return None

    plt = runtime_config.plt
    matplotlib = runtime_config.matplotlib # Assume loaded if available flag is true

    plot_data = []
    for model, data in results.items():
        if model.startswith("_"): continue
        summary = data.get("_summary")
        if summary and summary.get("status") == "Completed":
            value = summary.get(metric_key)
            if value is not None and value == value: # Check for valid number (not None or NaN)
                try:
                    model_name = summary.get("model_name", model)
                    stage = get_model_stage(model_name) # Always determine stage if grouping is enabled
                    plot_data.append((model_name, float(value), stage))
                except (ValueError, TypeError):
                    print(f"[WARN] Skipping model '{model}' for plot '{title}': Metric '{metric_key}' value '{value}' is not numeric.")
                    continue

    if not plot_data:
        return None

    # Sort data primarily by metric value
    plot_data.sort(key=lambda item: item[1], reverse=not lower_is_better)

    models = [item[0] for item in plot_data]
    values = [item[1] for item in plot_data]
    stages = [item[2] for item in plot_data] # Get stages for color mapping

    try:
        fig_height = max(4, len(models) * 0.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        # Determine colors based on stage if grouping
        bar_colors = [STAGE_COLORS.get(stage, 'grey') for stage in stages] if group_by_stage else 'skyblue'

        # Create bars
        bars = ax.barh(models, values, color=bar_colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(ylabel)
        ax.set_ylabel("Model")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis() # Display highest rank at the top

        # Add value labels to bars
        ax.bar_label(bars, fmt=f'{value_format}{"%" if is_percentage else ""}', padding=5, fontsize=9)

        # Adjust x-axis limits for padding
        if values:
            max_val = max(values)
            min_val = min(values)
            padding = (max_val - min_val) * 0.15 if max_val != min_val else abs(max_val) * 0.15 + 1
            if lower_is_better:
                # Ensure min_val is calculated correctly when lower is better
                min_val_for_limit = min(v for _, v, _ in plot_data)
                max_val_for_limit = max(v for _, v, _ in plot_data)
                padding = (max_val_for_limit - min_val_for_limit) * 0.15 if max_val_for_limit != min_val_for_limit else abs(max_val_for_limit) * 0.15 + 1
                ax.set_xlim(left=min_val_for_limit - padding, right=max_val_for_limit + padding * 0.1)
            else:
                ax.set_xlim(left=min_val - padding * 0.1, right=max_val + padding)

        # Add legend if grouping by stage
        if group_by_stage:
            handles = [plt.Rectangle((0,0),1,1, color=STAGE_COLORS[stage]) for stage in STAGES]
            ax.legend(handles, STAGES, title="Prompt Stage", loc='lower right', fontsize=9, title_fontsize=10)

        # Improve layout and grid
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout(pad=1.5)

        # Ensure image directory exists and save plot
        img_dir = runtime_config.report_img_dir
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / output_filename
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig) # Close the figure to free memory
        print(f"[INFO] Saved plot: {save_path}")
        # Return relative path for HTML report
        return f"images/{output_filename}" # Relative path from report HTML

    except Exception as e:
        print(f"[ERROR] Failed to generate plot '{title}': {e}")
        traceback.print_exc()
        if 'fig' in locals(): plt.close(fig) # Attempt to close figure even on error
        return None

# --- HTML Generation ---

def generate_html_report(results, benchmark_name, plot_paths, runtime_config):
    """Generates the full HTML report content from benchmark results."""
    # Args: results (dict): Benchmark data. benchmark_name (str): Title for the report. plot_paths (dict): Paths to generated plots. runtime_config (RuntimeConfig): Config object.
    # Returns: str | None: The generated HTML content as a string, or None on failure.

    template_file = runtime_config.html_template_file
    if not template_file.is_file():
        print(f"[ERROR] HTML template '{template_file}' not found.")
        return f"<html><body><h1>Error</h1><p>HTML template file not found at {template_file}</p></body></html>"

    try:
        with open(template_file, "r", encoding="utf-8") as f:
            template_html = f.read()
    except Exception as e:
         print(f"[ERROR] Failed to read HTML template '{template_file}': {e}")
         return f"<html><body><h1>Error</h1><p>Error reading template: {e}</p></body></html>"

    # --- Prepare Data for Template ---

    # 1. Summary Table Rows (Sort by Overall Score, then Ollama Score)
    summary_rows_html = []
    valid_models = [m for m, d in results.items() if not m.startswith("_") and isinstance(d, dict) and d.get("_summary", {}).get("status") == "Completed"]

    # Sort models: Higher Overall Score is better, higher Ollama Score is better tie-breaker
    sorted_models = sorted(valid_models,
                           key=lambda m: (results[m]["_summary"].get("overall_weighted_score", -1),
                                          results[m]["_summary"].get("ollama_perf_score", -1)),
                           reverse=True)

    for rank, model_name in enumerate(sorted_models, 1):
        summary = results[model_name]["_summary"]
        row = f"""<tr>
            <td>{rank}</td>
            <td>{summary.get('model_name', model_name)}</td>
            <td>{format_na(summary.get('overall_weighted_score'), precision=1)}</td>
            <td>{format_na(summary.get('ollama_perf_score'), precision=1)}</td>
            <td>{format_na(summary.get('accuracy'), suffix='%', precision=1)}</td>
            <td>{format_na(summary.get('tokens_per_sec_avg'), precision=1)}</td>
            <td>{format_na(summary.get('delta_ram_mb'), suffix=' MB', precision=1)}</td>
            <td>{format_na(summary.get('delta_gpu_mem_mb'), suffix=' MB', precision=1)}</td>
            <td>{format_na(summary.get('partial_score_avg'), suffix='%', precision=1)}</td>
            <td>{format_na(summary.get('avg_time_per_task'), suffix='s', precision=2)}</td>
            <td>{format_na(summary.get('error_rate'), suffix='%', precision=1)}</td>
        </tr>"""
        summary_rows_html.append(row)

    # Handle models that didn't complete or were skipped
    skipped_models = [m for m in results if not m.startswith("_") and isinstance(results[m], dict) and m not in valid_models]
    for model_name in skipped_models:
         summary = results[model_name].get("_summary", {"status": "Unknown"})
         row = f"""<tr>
             <td>-</td>
             <td>{summary.get('model_name', model_name)}</td>
             <td colspan="9" style="text-align:center; font-style:italic;">{summary.get('status')}</td>
         </tr>"""
         summary_rows_html.append(row)


    # 2. Per Task Type Aggregated Table Rows
    per_type_summary = defaultdict(lambda: {
        'model_count': 0, 'task_count_total': 0, 'correct_total': 0, 'api_errors_total': 0,
        'time_total': 0.0, 'score_sum_total': 0.0, 'scored_tasks_total': 0,
        'tokps_sum_total': 0.0, 'tokps_tasks_total': 0, 'success_api_calls_total': 0
    })

    for model in valid_models: # Only aggregate from completed models
        summary = results[model]["_summary"]
        has_ollama_metrics = summary.get("provider") == "ollama" # Only count tok/s for ollama

        for ttype, stats in summary.get("per_type", {}).items():
             agg = per_type_summary[ttype]
             agg['model_count'] += 1
             agg['task_count_total'] += stats.get('count', 0)
             agg['correct_total'] += stats.get('correct', 0)
             agg['api_errors_total'] += stats.get('api_errors', 0)
             agg['time_total'] += stats.get('time_sum', 0.0) # Use sum directly
             agg['success_api_calls_total'] += stats.get('success_api_calls', 0)

             if stats.get('avg_score') is not None and stats.get('score_count', 0) > 0:
                 # Ensure avg_score is float before multiplying
                 try:
                     avg_score_val = float(stats['avg_score'])
                     score_count_val = int(stats['score_count'])
                     agg['score_sum_total'] += avg_score_val * score_count_val
                     agg['scored_tasks_total'] += score_count_val
                 except (ValueError, TypeError):
                     print(f"[WARN] Invalid score data for {model}/{ttype}: avg_score={stats.get('avg_score')}, score_count={stats.get('score_count')}")

             if has_ollama_metrics and stats.get('avg_tokps') is not None and stats.get('tokps_count', 0) > 0:
                 # Ensure avg_tokps is float before multiplying
                 try:
                     avg_tokps_val = float(stats['avg_tokps'])
                     tokps_count_val = int(stats['tokps_count'])
                     agg['tokps_sum_total'] += avg_tokps_val * tokps_count_val
                     agg['tokps_tasks_total'] += tokps_count_val
                 except (ValueError, TypeError):
                     print(f"[WARN] Invalid tokps data for {model}/{ttype}: avg_tokps={stats.get('avg_tokps')}, tokps_count={stats.get('tokps_count')}")


    per_type_rows_html = []
    sorted_types = sorted(per_type_summary.keys())
    for ttype in sorted_types:
        agg = per_type_summary[ttype]
        if agg['task_count_total'] == 0: continue

        avg_acc = (agg['correct_total'] / agg['success_api_calls_total']) * 100 if agg['success_api_calls_total'] > 0 else 0.0
        avg_score = agg['score_sum_total'] / agg['scored_tasks_total'] if agg['scored_tasks_total'] > 0 else None
        avg_time = agg['time_total'] / agg['task_count_total']
        avg_tokps = agg['tokps_sum_total'] / agg['tokps_tasks_total'] if agg['tokps_tasks_total'] > 0 else None

        row = f"""<tr>
            <td>{ttype}</td>
            <td>{agg['task_count_total']} (across {agg['model_count']} models)</td>
            <td>{format_na(avg_acc, suffix='%', precision=1)}</td>
            <td>{format_na(avg_score, suffix='%', precision=1)}</td>
            <td>{format_na(avg_time, suffix='s', precision=2)}</td>
            <td>{format_na(avg_tokps, precision=1)}</td>
            <td>{agg['api_errors_total']}</td>
        </tr>"""
        per_type_rows_html.append(row)


    # 3. Detailed Results HTML Blocks (One collapsible per model)
    detailed_html_blocks = []
    task_definitions = results.get("_task_definitions", {})

    # Use the same sorted order as the summary table for consistency
    all_model_names_sorted = sorted_models + skipped_models

    for model_name in all_model_names_sorted:
        model_data = results[model_name]
        summary = model_data.get("_summary", {})
        status = summary.get("status", "Unknown")
        provider = summary.get("provider", "N/A")

        # Start collapsible block
        model_block = f'\n<button type="button" class="collapsible">Model: {model_name} ({provider}) - Status: {status}</button>\n'
        model_block += '<div class="content">\n'

        if status != "Completed":
            model_block += f"<p><i>Model run status: {status}. No detailed results available.</i></p>"
        else:
            # Add mini-summary table at the top of the details
            model_block += "<h4>Model Quick Stats:</h4>\n<table class='mini-summary'><tbody>\n"
            model_block += f"<tr><th>Overall Score</th><td>{format_na(summary.get('overall_weighted_score'), '', 1)}</td></tr>\n"
            if provider == "ollama": # Show Ollama score only for Ollama models
                 model_block += f"<tr><th>Ollama Score</th><td>{format_na(summary.get('ollama_perf_score'), '', 1)}</td></tr>\n"
            model_block += f"<tr><th>Accuracy</th><td>{format_na(summary.get('accuracy'), '%', 1)}</td></tr>\n"
            if provider == "ollama":
                model_block += f"<tr><th>Avg Tok/s</th><td>{format_na(summary.get('tokens_per_sec_avg'), '', 1)}</td></tr>\n"
                if summary.get("delta_ram_mb") is not None: model_block += f"<tr><th>RAM Delta</th><td>{format_na(summary.get('delta_ram_mb'), ' MB', 1)}</td></tr>\n"
                if summary.get("delta_gpu_mem_mb") is not None: model_block += f"<tr><th>GPU Mem Delta</th><td>{format_na(summary.get('delta_gpu_mem_mb'), ' MB', 1)}</td></tr>\n"
            model_block += f"<tr><th>Avg Time/Task</th><td>{format_na(summary.get('avg_time_per_task'), 's', 2)}</td></tr>\n"
            model_block += f"<tr><th>API Errors</th><td>{summary.get('api_errors', 0)}</td></tr>\n"
            model_block += "</tbody></table>\n"

            # Detailed task table
            model_block += "<h4 style='clear: both;'>Task Breakdown:</h4>\n" # Clear float
            model_block += "<table class='detailed-tasks'>\n<thead><tr><th>Task Name</th><th>Type</th><th>Prompt Snippet</th><th>Status</th><th>Metric</th><th>Details / Response Snippet</th><th>Duration (s)</th><th>Tok/s</th></tr></thead>\n<tbody>\n"

            # Ensure tasks are ordered consistently if possible (using task_definitions order)
            task_names_in_model = [tname for tname in model_data if not tname.startswith("_")]
            ordered_task_names = [tname for tname in task_definitions if tname in task_names_in_model]
            # Add any tasks run but not in current definitions (e.g., from cache)
            ordered_task_names += [tname for tname in task_names_in_model if tname not in ordered_task_names]

            for task_name in ordered_task_names:
                task_result = model_data[task_name]
                task_def = task_definitions.get(task_name, {}) # Get definition if available

                prompt_snippet = truncate_text(task_def.get("prompt", "N/A"), 100)
                response_snippet = truncate_text(task_result.get("response", ""), 150)
                details = task_result.get("details", "N/A")
                metric = task_result.get("metric")
                ttype = task_result.get("task_type", task_def.get("type", "N/A")) # Use result's type first
                error = task_result.get("error")

                status_str = "N/A"
                metric_str = ""
                status_class = "status-unknown" # CSS class

                if error:
                    status_str = "API Error"
                    metric_str = "N/A"
                    details = error # Show full API error message here
                    response_snippet = "N/A"
                    status_class = "status-error"
                elif isinstance(metric, bool):
                    status_str = "PASS" if metric else "FAIL"
                    metric_str = str(metric)
                    status_class = "status-pass" if metric else "status-fail"
                elif isinstance(metric, (float, int)):
                    score = float(metric)
                    pass_threshold = runtime_config.passing_score_threshold
                    is_pass = (score >= pass_threshold)
                    status_str = "PASS" if is_pass else "FAIL"
                    metric_str = f"{score:.1f}%"
                    status_class = "status-pass" if is_pass else "status-fail"
                else: # Unknown metric type or None metric without error
                    status_str = "Unknown" if metric is None else "Invalid Metric"
                    metric_str = str(metric) if metric is not None else "N/A"
                    status_class = "status-unknown"

                # Combine details and response snippet for display
                combined_details = f"<i>Details:</i> {details}" if details and details != "N/A" else ""
                if response_snippet and response_snippet != "N/A":
                     combined_details += f"<br><i>Resp:</i> <pre class='response-snippet'>{response_snippet}</pre>"
                elif error:
                     combined_details = f"<i>Error:</i> <pre class='response-snippet'>{details}</pre>"


                row = f"""<tr>
                    <td>{task_name}</td>
                    <td>{ttype}</td>
                    <td><pre class='prompt-snippet'>{prompt_snippet}</pre></td>
                    <td class='{status_class}'>{status_str}</td>
                    <td>{metric_str}</td>
                    <td class='details-cell'>{combined_details}</td>
                    <td>{format_na(task_result.get('duration'), precision=2)}</td>
                    <td>{format_na(task_result.get('tokens_per_sec'), precision=1)}</td>
                </tr>"""
                model_block += row

            model_block += "</tbody>\n</table>\n"
        model_block += "</div>\n" # End content div
        detailed_html_blocks.append(model_block)


    # 4. Full JSON Data
    # Exclude helper keys added during run/reporting
    results_for_json = {k: v for k, v in results.items() if k not in ["_task_definitions", "_task_categories", "_category_weights_used", "_ollama_score_weights_used"]}
    try:
        full_json = json.dumps(results_for_json, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Could not serialize full results to JSON for report: {e}")
        full_json = f"Error serializing results: {e}"


    # --- Inject data into template ---
    final_html = template_html
    final_html = final_html.replace("{{REPORT_TITLE}}", benchmark_name)
    final_html = final_html.replace("{{TIMESTAMP}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"))

    # Inject plot paths (use placeholders if path is None or generation failed)
    placeholder_img_rel_path = "images/placeholder.png" # Assume a placeholder exists
    plot_html_content = ""
    plot_keys = [
        ("Overall Score", "overall_score", "Overall Weighted Score Ranking (Higher is Better)"),
        ("Ollama Performance Score", "ollama_score", "Ollama Performance Score Ranking (Higher is Better)"),
        ("Accuracy", "accuracy", "Overall Accuracy (%) Ranking (Higher is Better)"),
        ("Tokens/Second (Ollama)", "tokps", "Average Tokens/Second Ranking (Ollama Models, Higher is Better)"),
        ("RAM Delta (Ollama)", "ram", "Peak RAM Increase Ranking (Ollama Models, Lower is Better)"),
        ("GPU Memory Delta (Ollama)", "gpu", "Peak GPU Memory Increase Ranking (Ollama Models - GPU 0, Lower is Better)"),
        ("Accuracy by Stage", "accuracy_by_stage", "Accuracy by Stage (None, Gates, Persona)"),
        ("Overall Score by Stage", "overall_score_by_stage", "Overall Score by Stage (None, Gates, Persona)"),
    ]

    for plot_title, plot_key, plot_tooltip in plot_keys:
        img_path = plot_paths.get(plot_key)
        if img_path and pathlib.Path(runtime_config.report_dir / img_path).is_file():
            plot_html_content += f'<img src="{img_path}" alt="{plot_title} Plot" title="{plot_tooltip}">\n'
            if plot_key != plot_keys[-1][1]: # Don't add separator after the last plot
                 plot_html_content += '<hr class="plot-separator">\n'
        elif runtime_config.visualizations_enabled and runtime_config.matplotlib_available:
            # Plot was expected but failed or data was missing
            print(f"[WARN] Plot '{plot_title}' path not found or file missing, skipping in report.")
            # Optionally add a placeholder text in the report here
            # plot_html_content += f'<p style="font-style: italic; color: grey;">Plot for {plot_title} could not be generated.</p><hr class="plot-separator">\n'


    # Replace the placeholder comment with the generated plot images
    final_html = final_html.replace("<!-- Plots injected here -->", plot_html_content if plot_html_content else "<p>No plots were generated.</p>")

    # Inject table rows and detail blocks
    final_html = final_html.replace("{{SUMMARY_TABLE_ROWS}}", "\n".join(summary_rows_html))
    final_html = final_html.replace("{{PER_TYPE_TABLE_ROWS}}", "\n".join(per_type_rows_html))
    final_html = final_html.replace("{{DETAILED_RESULTS_HTML}}", "\n".join(detailed_html_blocks))
    final_html = final_html.replace("{{FULL_RESULTS_JSON}}", full_json)

    # Add warning if plots were expected but library was missing
    if runtime_config.visualizations_enabled and not runtime_config.matplotlib_available:
         warning_msg = '<p style="color:red; text-align:center; font-style:italic;">Note: Matplotlib library not found, plots could not be generated. Install with: pip install matplotlib</p>'
         # Inject warning before the plot container
         final_html = re.sub(r'(<div class="plot-container">)', warning_msg + r'\1', final_html, count=1)


    return final_html


# --- File Operations ---

def save_report(html_content, runtime_config):
    """Saves the generated HTML report to the configured file path."""
    # Args: html_content (str): The HTML string. runtime_config (RuntimeConfig): Config object with paths.
    # Returns: pathlib.Path | None: The path where the report was saved, or None on failure.
    report_file = runtime_config.report_html_file
    try:
        # Ensure parent directory exists
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[INFO] Report saved to: {report_file.resolve()}")
        return report_file
    except Exception as e:
        print(f"[ERROR] Failed to save HTML report to {report_file}: {e}")
        return None

def open_report_auto(report_path):
    """Attempts to open the saved report in the default web browser."""
    # Args: report_path (pathlib.Path): Path to the HTML report file.
    # Returns: None
    if not report_path: return
    try:
        report_uri = report_path.resolve().as_uri()
        print(f"[INFO] Attempting to open report: {report_uri}")
        webbrowser.open(report_uri)
    except Exception as e:
        print(f"[WARN] Could not auto-open report: {e}\n      Report is at: {report_path.resolve()}")