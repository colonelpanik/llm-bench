#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- LLM Benchmark Runner CLI ---

import argparse
import sys
import os
import pathlib
import time
import json
import traceback
import platform # For dependency check
from collections import defaultdict

# --- Import Configuration and Core Modules ---
import config # Base config and loading function
from config import RuntimeConfig, load_config_from_file # Import the class and loader
from utils import format_na # Import utils early if needed for printing
from llm_clients import pull_ollama_model, get_local_ollama_models # Needed for pre-checks
from benchmark_runner import run_benchmark_set
from scoring import compute_performance_scores
from reporting import generate_ranking_plot, generate_html_report, save_report, open_report_auto
from cache_manager import get_cache_path, load_cache, save_cache, clear_cache
from export import export_summary_csv, export_details_json # Added for export
# Import system_monitor here if check_dependencies needs it directly
import system_monitor

# --- Dependency Check Function ---
# (check_dependencies function remains the same as before)
def check_dependencies(check_runtime_config):
    """Checks for optional dependencies and reports their status."""
    print("\n--- Dependency Check ---")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} ({platform.release()})")
    all_ok = True
    lib_status = {}

    # Check PyYAML (Required for config, needed for YAML tasks)
    try:
        import yaml
        print("[ OK ] PyYAML: Found")
        lib_status['yaml'] = True
    except ImportError:
        print("[FAIL] PyYAML: Not Found (Required for YAML config/tasks. Install: pip install PyYAML)")
        all_ok = False
        lib_status['yaml'] = False

    # Check psutil (RAM Monitor)
    try:
        import psutil
        print("[ OK ] psutil: Found")
        # Test getting PIDs (might fail due to permissions, but confirms import)
        try:
            # Pass the runtime_config needed for API fallback in get_ollama_pids
            pids = system_monitor.get_ollama_pids(psutil, check_runtime_config)
            print(f"       - psutil check: Found {len(pids)} potential Ollama PIDs (permissions permitting)")
            lib_status['psutil'] = True
        except Exception as e:
            print(f"       - psutil check: Error during PID check (permissions?): {e}")
            lib_status['psutil'] = 'Error' # Mark as error state
    except ImportError:
        print("[WARN] psutil: Not Found (Required for --ram-monitor enable. Install: pip install psutil)")
        lib_status['psutil'] = False

    # Check pynvml (GPU Monitor)
    try:
        import pynvml
        print("[ OK ] PyNVML: Found")
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            print(f"       - NVML Initialized: OK. Found {count} NVIDIA GPU(s).")
            if count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"       - GPU 0 Memory Info: OK (Used: {mem_info.used / 1024**2:.1f} MiB)")
                lib_status['pynvml'] = True
            else:
                print("       - GPU Monitoring: Inactive (No NVIDIA GPUs detected).")
                lib_status['pynvml'] = 'No GPU'
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            print(f"       - NVML Initialization Error: {e}. GPU Monitoring disabled.")
            lib_status['pynvml'] = 'NVML Error'
            # Treat NVML error as critical only if GPU monitoring is explicitly enabled in config?
            # For check, just report it.
        except Exception as e:
            print(f"       - Unexpected PyNVML Error: {e}")
            lib_status['pynvml'] = 'Error'
    except ImportError:
        print("[WARN] PyNVML: Not Found (Required for --gpu-monitor enable. Install: pip install pynvml)")
        lib_status['pynvml'] = False

    # Check Matplotlib (Visualizations)
    try:
        import matplotlib
        matplotlib.use('Agg') # Attempt to set backend early
        import matplotlib.pyplot as plt
        print("[ OK ] Matplotlib: Found")
        lib_status['matplotlib'] = True
    except ImportError:
        print("[WARN] Matplotlib: Not Found (Required for --visualizations enable. Install: pip install matplotlib)")
        lib_status['matplotlib'] = False
    except Exception as e: # Catch backend errors etc.
        print(f"[WARN] Matplotlib: Error during import/backend setting: {e}. Visualizations may fail.")
        lib_status['matplotlib'] = 'Error'


    # Check Sentence Transformers (Semantic Eval)
    try:
        from sentence_transformers import SentenceTransformer
        print("[ OK ] sentence-transformers: Found")
        model_name = 'all-MiniLM-L6-v2' # Standard model for check
        print(f"       - Attempting to load semantic model '{model_name}'...")
        try:
            # Use a minimal load check
            semantic_model = SentenceTransformer(model_name)
            _ = semantic_model.encode("test sentence")
            print(f"       - Semantic model loaded and test encode successful: OK")
            lib_status['sentence-transformers'] = True
            del semantic_model # Clean up
        except Exception as e:
            print(f"       - Semantic model load/test FAILED: {e}. Semantic evaluation disabled if enabled.")
            lib_status['sentence-transformers'] = 'Model Error'
            # Treat model error as critical only if semantic eval is enabled?
            # For check, just report it.
    except ImportError:
        print("[WARN] sentence-transformers: Not Found (Required for --semantic-eval enable. Install: pip install sentence-transformers)")
        lib_status['sentence-transformers'] = False

    print("\n--- Summary ---")
    if all_ok and all(v not in ['Error', 'NVML Error', 'Model Error'] for v in lib_status.values()):
        print("Core dependencies (requests, PyYAML) present.")
        print("All checked optional features seem ready based on installed libraries and basic checks.")
        print("Note: Runtime errors (e.g., permissions, specific hardware issues) can still occur.")
        return True
    else:
        print("Potential issues detected with core or optional dependencies. Check messages above.")
        print("Some features might be unavailable or encounter errors during the benchmark.")
        return False


# --- Function to Setup Runtime Configuration ---
def setup_runtime_config(args, loaded_file_config):
    """Initializes RuntimeConfig, applies file config, env vars, CLI args, and performs conditional imports."""

    cfg = RuntimeConfig() # Initialize with base defaults

    # 1. Apply settings from the loaded config file
    cfg.update_from_file_config(loaded_file_config)

    # 2. Apply Environment Variables (before CLI args)
    cfg.update_from_environment()

    # 3. Apply CLI arguments (these override file & ENV settings)

    # --- Model Selection Logic ---
    if args.test_model:
        # --test-model overrides default_models from config
        cfg.models_to_benchmark = sorted(list(set(args.test_model)))
        if args.code_model:
            # If --code-model is ALSO provided, use it specifically for code tasks
            cfg.code_models_to_benchmark = sorted(list(set(args.code_model)))
        else:
            # If ONLY --test-model is provided, it applies to ALL tasks, overriding config code_models
            cfg.code_models_to_benchmark = list(cfg.models_to_benchmark)
            print("[INFO] --test-model provided without --code-model. Applying specified models to all task types.")
    else:
        # No --test-model provided, use defaults from config (already loaded)
        # If --code-model is provided, it overrides config code_models
        if args.code_model:
            cfg.code_models_to_benchmark = sorted(list(set(args.code_model)))
        else:
            # No --code-model provided either, use config code_models (already loaded)
            # If config also didn't have code_models, default code list to general list
            if not cfg.code_models_to_benchmark:
                 cfg.code_models_to_benchmark = list(cfg.models_to_benchmark)


    # --- Other CLI Overrides ---
    # Paths override
    if args.tasks_file: cfg.tasks_file = args.tasks_file
    if args.report_dir: cfg.report_dir = args.report_dir
    if args.cache_dir: cfg.cache_dir = args.cache_dir
    if args.template_file: cfg.html_template_file = args.template_file

    # Recalculate derived paths after potential overrides
    cfg.report_img_dir = cfg.report_dir / "images"
    cfg.report_html_file = cfg.report_dir / "report.html"

    # Execution control overrides
    if args.retries is not None: cfg.max_retries = args.retries
    if args.retry_delay is not None: cfg.retry_delay = args.retry_delay

    # API Key: CLI > ENV > File Config
    if args.gemini_key: # CLI highest precedence
        cfg.gemini_key = args.gemini_key
        print("[INFO] Using Gemini API Key from --gemini-key argument.")
    # ENV/File precedence handled earlier

    cfg.verbose = args.verbose

    # Category weights: CLI overrides file/ENV config
    if args.category_weights:
        try:
            cli_weights = json.loads(args.category_weights)
            if not isinstance(cli_weights, dict): raise ValueError
            print(f"[INFO] Using category weights from --category-weights argument: {cli_weights}")
            cfg.category_weights = cli_weights
            cfg.default_category_weight = cli_weights.get('default', cfg.default_category_weight)
        except (json.JSONDecodeError, ValueError):
             print(f"[WARN] Invalid JSON in --category-weights argument. Using weights from config file or defaults.")
    # cfg.category_weights holds value from ENV/file if CLI is invalid/not provided

    # Feature toggles: CLI 'enable'/'disable' overrides file/ENV config
    if args.ram_monitor is not None: cfg.ram_monitor_enabled = (args.ram_monitor == "enable")
    if args.gpu_monitor is not None: cfg.gpu_monitor_enabled = (args.gpu_monitor == "enable")
    if args.visualizations is not None: cfg.visualizations_enabled = (args.visualizations == "enable")
    if args.semantic_eval is not None: cfg.semantic_eval_enabled = (args.semantic_eval == "enable")

    # --- 4. Perform Conditional Imports based on FINALIZED config ---
    print("\n--- Initializing Optional Dependencies Based on Config ---")
    # (Conditional import logic remains the same...)
    if cfg.ram_monitor_enabled:
        try:
            import psutil
            cfg.psutil = psutil
            cfg.psutil_available = True
            print("[INFO] psutil imported successfully. RAM monitoring enabled.")
        except ImportError:
            print("[WARN] psutil not found, RAM monitoring disabled. Run: pip install psutil")
            cfg.ram_monitor_enabled = False
    else: print("[INFO] RAM monitoring disabled by config/argument.")

    try:
        import yaml
        cfg.yaml = yaml
        cfg.pyyaml_available = True
    except ImportError:
        print("[WARN] PyYAML not found. YAML config loading failed, and YAML tasks will be skipped. Run: pip install PyYAML")
        cfg.pyyaml_available = False

    if cfg.visualizations_enabled:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            cfg.matplotlib = matplotlib
            cfg.plt = plt
            cfg.mticker = mticker
            cfg.matplotlib_available = True
            print("[INFO] Matplotlib imported successfully. Visualizations enabled.")
        except ImportError:
            print("[WARN] Matplotlib not found, visualizations disabled. Run: pip install matplotlib")
            cfg.visualizations_enabled = False
            cfg.matplotlib_available = False
        except Exception as e:
            print(f"[WARN] Matplotlib import or backend setting failed: {e}. Visualizations disabled.")
            cfg.visualizations_enabled = False
            cfg.matplotlib_available = False
    else: print("[INFO] Visualizations disabled by config/argument.")

    if cfg.semantic_eval_enabled:
        try:
            from sentence_transformers import SentenceTransformer, util
            try:
                model_name = 'all-MiniLM-L6-v2'
                print(f"[INFO] Loading semantic model '{model_name}'...")
                semantic_model = SentenceTransformer(model_name)
                _ = semantic_model.encode("warmup")
                cfg.SentenceTransformer = SentenceTransformer
                cfg.st_util = util
                cfg.semantic_model = semantic_model
                cfg.sentence_transformers_available = True
                print("[INFO] Sentence Transformers imported and model loaded. Semantic evaluation enabled.")
            except Exception as model_load_e:
                print(f"[WARN] Sentence Transformers library found, but failed to load model '{model_name}': {model_load_e}. Semantic evaluation disabled.")
                cfg.sentence_transformers_available = False
                cfg.semantic_eval_enabled = False
        except ImportError:
            print("[WARN] sentence-transformers not found, semantic evaluation disabled. Run: pip install sentence-transformers")
            cfg.semantic_eval_enabled = False
            cfg.sentence_transformers_available = False
    else: print("[INFO] Semantic evaluation disabled by config/argument.")

    if cfg.gpu_monitor_enabled:
        try:
            import pynvml
            cfg.pynvml = pynvml
            try:
                pynvml.nvmlInit()
                cfg.gpu_count = pynvml.nvmlDeviceGetCount()
                if cfg.gpu_count > 0:
                    cfg.pynvml_available = True
                    print(f"[INFO] PyNVML imported and initialized. Found {cfg.gpu_count} NVIDIA GPU(s). GPU monitoring enabled.")
                else:
                    print("[INFO] PyNVML found, but no NVIDIA GPUs detected. GPU monitoring disabled.")
                    cfg.pynvml_available = False
                    cfg.gpu_monitor_enabled = False
            except pynvml.NVMLError as nvml_err:
                print(f"[WARN] PyNVML found but failed to initialize: {nvml_err}. GPU monitoring disabled.")
                cfg.pynvml_available = False
                cfg.gpu_monitor_enabled = False
            except Exception as init_err:
                 print(f"[WARN] Unexpected error during PyNVML init: {init_err}. GPU monitoring disabled.")
                 cfg.pynvml_available = False
                 cfg.gpu_monitor_enabled = False
        except ImportError:
            print("[WARN] pynvml not found, GPU monitoring disabled. Run: pip install pynvml")
            cfg.gpu_monitor_enabled = False
            cfg.pynvml_available = False
    else: print("[INFO] GPU monitoring disabled by config/argument.")

    return cfg


# --- Function to Load and Validate Tasks ---
# (load_and_validate_tasks function remains the same as before)
def load_and_validate_tasks(tasks_file, runtime_config):
    """Loads tasks from JSON file, validates structure, and filters based on dependencies."""
    loaded_tasks_dict = {} # name -> task_def map
    task_categories_map = defaultdict(list) # category -> [task_names]

    try:
        print(f"[INFO] Loading tasks from: {tasks_file.resolve()}")
        if not tasks_file.is_file():
            raise FileNotFoundError(f"Tasks file '{tasks_file}' not found.")

        with open(tasks_file, 'r', encoding='utf-8') as f:
            tasks_json = json.load(f)

        if not isinstance(tasks_json, dict):
            raise ValueError("Tasks JSON root must be a dictionary (e.g., category names as keys).")

        validated_count = 0
        skipped_dependency = 0
        task_names = set()

        for category, tasks_in_category in tasks_json.items():
            if not isinstance(tasks_in_category, list):
                print(f"[WARN] Category '{category}' in tasks file is not a list. Skipping.")
                continue

            for idx, task in enumerate(tasks_in_category):
                 # Basic structure validation
                 if not isinstance(task, dict): raise ValueError(f"Task #{idx+1} in category '{category}' is not an object.")
                 if not all(k in task for k in ("name", "type", "prompt")): raise ValueError(f"Task #{idx+1} in '{category}' missing required keys (name, type, prompt).")
                 task_name = task["name"]
                 if not isinstance(task_name, str) or not task_name.strip(): raise ValueError(f"Task #{idx+1} in '{category}' has invalid/empty name.")
                 if task_name in task_names: raise ValueError(f"Duplicate task name found: '{task_name}'. Names must be unique across all categories.")
                 task_names.add(task_name)
                 task_type = task.get("type")
                 if not isinstance(task_type, str) or not task_type.strip(): raise ValueError(f"Task '{task_name}' has invalid/empty type.")

                 # Dependency checks
                 skip_reason = None
                 eval_method = task.get("evaluation_method")
                 if eval_method == "semantic" and not runtime_config.semantic_eval_enabled:
                     skip_reason = "Semantic evaluation disabled by config/argument (--semantic-eval disable)"
                 elif eval_method == "semantic" and not runtime_config.sentence_transformers_available:
                     skip_reason = "Semantic evaluation library (sentence-transformers) or model unavailable"
                 elif task_type.startswith("yaml_") and not runtime_config.pyyaml_available:
                     skip_reason = "YAML task requires PyYAML library, which is not available"

                 if skip_reason:
                     print(f"[INFO] Skipping task '{task_name}': {skip_reason}.")
                     skipped_dependency += 1
                     continue # Skip this task

                 # Task seems valid and runnable
                 loaded_tasks_dict[task_name] = task
                 task_categories_map[category].append(task_name)
                 validated_count += 1

        if not loaded_tasks_dict:
            raise ValueError("No valid or runnable tasks found after validation and dependency checks.")

        print(f"[SUCCESS] Loaded and validated {validated_count} runnable tasks ({skipped_dependency} skipped due to dependencies).")
        return loaded_tasks_dict, dict(task_categories_map) # Return as plain dict

    except FileNotFoundError as e: print(f"[ERROR] {e}"); return None, None
    except (json.JSONDecodeError, ValueError, TypeError) as e: print(f"[ERROR] Task file format error: {type(e).__name__}: {e}"); return None, None
    except Exception as e: print(f"[ERROR] Unexpected error loading tasks: {e}"); traceback.print_exc(); return None, None


# --- Main Function ---
def main():
    # --- Argument Parsing ---
    # (Argument parser setup remains the same...)
    parser = argparse.ArgumentParser(
        description="LLM Benchmark Runner - Evaluate local and remote LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config-file", default=config.DEFAULT_CONFIG_FILE, type=pathlib.Path, help="Path to the YAML configuration file.")
    parser.add_argument("--test-model", action="append", help="Specify a model name to benchmark. Use multiple times. Overrides config file defaults for ALL tasks unless --code-model is also used.")
    parser.add_argument("--code-model", action="append", help="Specify model name ONLY for code tasks. Overrides config file 'code_models'. Use multiple times.")
    parser.add_argument("--task-set", choices=["all", "nlp", "code", "other"], default="all", help="Which category of tasks to run (uses category keys in tasks file).")
    parser.add_argument("--task-name", action="append", help="Run ONLY specific tasks by name. Use multiple times. Overrides --task-set.")
    parser.add_argument("--gemini-key", default=None, help="API key for Google Gemini. Overrides ENV var and config file.")
    parser.add_argument("--pull-ollama-models", action="store_true", help="Attempt to pull Ollama models via API if not found locally.")
    parser.add_argument("--benchmark-name", default="LLM Benchmark Run", help="Name for this run (used for cache file and report title).")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache and run fresh.")
    parser.add_argument("--clear-cache", action="store_true", help="Delete the cache file before running.")
    parser.add_argument("--tasks-file", default=None, type=pathlib.Path, help="Path to tasks JSON definition file. Overrides config.")
    parser.add_argument("--report-dir", default=None, type=pathlib.Path, help="Directory to save report and images. Overrides config.")
    parser.add_argument("--cache-dir", default=None, type=pathlib.Path, help="Directory to store cache files. Overrides config.")
    parser.add_argument("--template-file", default=None, type=pathlib.Path, help="Path to the HTML report template file. Overrides config.")
    parser.add_argument("--retries", type=int, default=None, help="Number of retries for transient API errors. Overrides config.")
    parser.add_argument("--retry-delay", type=int, default=None, help="Delay (seconds) between retries. Overrides config.")
    parser.add_argument("--category-weights", type=str, default=None, help="JSON string mapping categories to weights for scoring. Overrides config.")
    parser.add_argument("--ram-monitor", choices=["enable", "disable"], default=None, help="Enable/disable CPU RAM monitoring. Overrides config.")
    parser.add_argument("--gpu-monitor", choices=["enable", "disable"], default=None, help="Enable/disable NVIDIA GPU memory monitoring. Overrides config.")
    parser.add_argument("--visualizations", choices=["enable", "disable"], default=None, help="Enable/disable report plots. Overrides config.")
    parser.add_argument("--semantic-eval", choices=["enable", "disable"], default=None, help="Enable/disable semantic evaluation. Overrides config.")
    parser.add_argument("--open-report", action="store_true", help="Automatically open the generated HTML report in a web browser.")
    parser.add_argument("--export-summary-csv", action="store_true", help="Export summary results to a CSV file in the report directory.")
    parser.add_argument("--export-details-json", action="store_true", help="Export detailed task results (no summaries) to a JSON file.")
    parser.add_argument("--check-dependencies", action="store_true", help="Check optional dependencies and exit.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging during model queries.")


    try:
        args = parser.parse_args()
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"[ERROR] Error parsing arguments: {e}")
        sys.exit(1)

    # --- Load Configuration File ---
    print(f"Attempting to load configuration from: {args.config_file}")
    loaded_file_config = load_config_from_file(args.config_file)

    # --- Setup Runtime Configuration (Handles overrides and imports) ---
    runtime_config = setup_runtime_config(args, loaded_file_config)

    # --- Handle Utility Actions First ---
    if args.check_dependencies:
        check_dependencies(runtime_config)
        sys.exit(0)

    # --- Proceed with Benchmark ---
    print("\n--- Starting Benchmark ---")
    run_start_time = time.time()

    # --- Create Directories ---
    try:
        runtime_config.report_dir.mkdir(parents=True, exist_ok=True)
        runtime_config.report_img_dir.mkdir(parents=True, exist_ok=True)
        runtime_config.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using Report Directory: {runtime_config.report_dir.resolve()}")
        print(f"[INFO] Using Cache Directory: {runtime_config.cache_dir.resolve()}")
    except Exception as e:
        print(f"[ERROR] Could not create necessary directories: {e}")
        sys.exit(1)

    # --- Load and Validate Tasks ---
    all_tasks_dict, task_categories_map = load_and_validate_tasks(runtime_config.tasks_file, runtime_config)
    if all_tasks_dict is None:
        sys.exit(1)

    # --- Select Tasks Based on Arguments ---
    # (Task selection logic remains the same...)
    tasks_to_run_names = set()
    if args.task_name:
        requested_names = set(args.task_name)
        tasks_to_run_names = {name for name in requested_names if name in all_tasks_dict}
        not_found = requested_names - tasks_to_run_names
        if not_found:
            print(f"[WARN] Requested task names not found or not runnable: {sorted(list(not_found))}")
        if not tasks_to_run_names:
            print("[ERROR] No specified tasks found or runnable. Exiting.")
            sys.exit(1)
        print(f"[INFO] Selected {len(tasks_to_run_names)} tasks based on --task-name argument.")
    else:
        selected_task_set = args.task_set.lower()
        if selected_task_set == "all":
            tasks_to_run_names.update(all_tasks_dict.keys())
        else:
            category_key_map = {
                "nlp": "General NLP", "code": "Code Generation", "other": "Other"
            }
            target_category = category_key_map.get(selected_task_set)
            if target_category and target_category in task_categories_map:
                 tasks_to_run_names.update(task_categories_map[target_category])
            else:
                 print(f"[WARN] Task set '{args.task_set}' selected, but no runnable tasks found in category '{target_category or args.task_set.upper()}'.")
                 print("[INFO] Exiting. Check task set or tasks file.")
                 sys.exit(1)
        print(f"[INFO] Selected {len(tasks_to_run_names)} tasks based on task set '{args.task_set}'.")

    tasks_to_run = [all_tasks_dict[name] for name in tasks_to_run_names if name in all_tasks_dict]
    if not tasks_to_run:
         print(f"[ERROR] No tasks selected or runnable. Exiting.")
         sys.exit(1)

    # --- Determine Final List of Models to Run ---
    # These lists are now correctly populated by setup_runtime_config based on args
    model_list = runtime_config.models_to_benchmark # Models for general tasks
    code_model_list = runtime_config.code_models_to_benchmark # Models for code tasks
    all_models_to_run = sorted(list(set(model_list + code_model_list))) # All unique models needed

    print(f"[INFO] Models targeted for *General* tasks ({len(model_list)}): {model_list}")
    print(f"[INFO] Models targeted for *Code* tasks ({len(code_model_list)}): {code_model_list}")
    print(f"[INFO] All unique models requiring checks/pulls ({len(all_models_to_run)}): {all_models_to_run}")

    # --- Check Model Availability / Pull ---
    # (Model check logic remains the same, using all_models_to_run)
    gemini_models_in_list = [m for m in all_models_to_run if m.startswith("gemini-") or m.startswith("models/")]
    ollama_models_in_list = [m for m in all_models_to_run if not (m.startswith("gemini-") or m.startswith("models/"))]

    if gemini_models_in_list and not runtime_config.gemini_key:
        print(f"[WARN] Gemini models ({gemini_models_in_list}) selected, but no API key provided. These models will be skipped.")

    if ollama_models_in_list:
        print("[INFO] Checking availability of local Ollama models...")
        try:
            local_ollama_models = get_local_ollama_models(runtime_config)
            if local_ollama_models is None:
                 print("[ERROR] Could not connect to Ollama to check models. Ollama benchmarks will likely fail.")
            else:
                missing_ollama = [m for m in ollama_models_in_list if m not in local_ollama_models]
                if missing_ollama:
                    print(f"[WARN] Local Ollama instance is missing required models: {missing_ollama}")
                    if args.pull_ollama_models:
                        print("[INFO] Attempting to pull missing Ollama models (--pull-ollama-models enabled)...")
                        pulled_successfully = []
                        for model_to_pull in missing_ollama:
                            if pull_ollama_model(model_to_pull, runtime_config):
                                pulled_successfully.append(model_to_pull)
                            else:
                                print(f"[ERROR] Failed to pull model '{model_to_pull}'. It will be skipped if required.")
                            time.sleep(1)
                        if pulled_successfully: print(f"[INFO] Pull attempt finished for: {pulled_successfully}")
                    else:
                        print("[INFO] Missing models will be skipped. Use --pull-ollama-models to attempt download.")
                else:
                    print("[INFO] All required Ollama models found locally.")
        except Exception as e:
             print(f"[ERROR] Failed during Ollama model check: {e}")
             traceback.print_exc()


    # --- Cache Handling ---
    cache_file = get_cache_path(runtime_config.cache_dir, args.benchmark_name)
    cached_results = None
    if args.clear_cache:
        clear_cache(cache_file)
    if not args.no_cache:
        cached_results = load_cache(cache_file, runtime_config.cache_ttl)

    # --- Execute Benchmark ---
    # (Benchmark execution logic remains the same, using all_models_to_run)
    results = {}
    run_reason = ""
    if cached_results:
        print("[INFO] Using cached results.")
        results = cached_results
        run_reason = "Using cache"
        if "_task_definitions" not in results: results["_task_definitions"] = all_tasks_dict
        if "_task_categories" not in results: results["_task_categories"] = task_categories_map
        if "_category_weights_used" not in results: results["_category_weights_used"] = runtime_config.category_weights
        if "_ollama_score_weights_used" not in results: results["_ollama_score_weights_used"] = runtime_config.ollama_score_weights
        print("[INFO] Recomputing performance scores based on cached data and current weights...")
        compute_performance_scores(results, runtime_config.category_weights, runtime_config.default_category_weight)

    else:
        if args.no_cache: run_reason = "--no-cache specified"
        elif args.clear_cache: run_reason = "Cache cleared (--clear-cache)"
        else: run_reason = "No valid cache found or cache expired"
        print(f"[INFO] {run_reason}, running benchmark...")

        try:
            results = run_benchmark_set(
                benchmark_name=args.benchmark_name,
                model_list=all_models_to_run, # Pass the combined list
                tasks=tasks_to_run,
                task_categories_map=task_categories_map,
                runtime_config=runtime_config # Pass the full config object
            )

            if results and any(not k.startswith("_") and isinstance(v, dict) for k, v in results.items()):
                print("[INFO] Computing performance scores for new results...")
                compute_performance_scores(results, runtime_config.category_weights, runtime_config.default_category_weight)
                results["_task_definitions"] = all_tasks_dict
                results["_task_categories"] = task_categories_map
                results["_category_weights_used"] = runtime_config.category_weights
                results["_ollama_score_weights_used"] = runtime_config.ollama_score_weights
                if not args.no_cache:
                    save_cache(cache_file, results)
            else:
                print("[WARN] Benchmark run produced no valid model results. Skipping scoring and caching.")

        except KeyboardInterrupt:
            print("\n[WARN] Benchmark run interrupted by user (Ctrl+C).")
            results = {}
        except Exception as e:
            print(f"\n[ERROR] Unexpected error during benchmark execution: {e}")
            traceback.print_exc()
            sys.exit(1)


    # --- Generate Plots and Report ---
    # (Reporting and Export logic remains the same...)
    report_path = None
    if results and any(not k.startswith("_") and isinstance(v, dict) for k, v in results.items()):
        if "_task_definitions" not in results: results["_task_definitions"] = all_tasks_dict
        if "_task_categories" not in results: results["_task_categories"] = task_categories_map
        if "_category_weights_used" not in results: results["_category_weights_used"] = runtime_config.category_weights
        if "_ollama_score_weights_used" not in results: results["_ollama_score_weights_used"] = runtime_config.ollama_score_weights

        plot_paths = {}
        if runtime_config.visualizations_enabled:
            if runtime_config.matplotlib_available:
                print("[INFO] Generating report visualizations...")
                try:
                    plot_paths["overall_score"] = generate_ranking_plot(results, "overall_weighted_score", "Overall Score Ranking", "Score (0-100, higher better)", "overall_score_plot.png", runtime_config)
                    plot_paths["ollama_score"] = generate_ranking_plot(results, "ollama_perf_score", "Ollama Perf Score Ranking", "Score (0-100, higher better)", "ollama_score_plot.png", runtime_config)
                    plot_paths["accuracy"] = generate_ranking_plot(results, "accuracy", "Accuracy Ranking", "Accuracy (%)", "accuracy_plot.png", runtime_config, is_percentage=True)
                    plot_paths["tokps"] = generate_ranking_plot(results, "tokens_per_sec_avg", "Avg Tok/s Ranking (Ollama)", "Tokens/Second", "tokps_plot.png", runtime_config)
                    plot_paths["ram"] = generate_ranking_plot(results, "delta_ram_mb", "Peak RAM Delta Ranking (Ollama)", "RAM Delta (MB, lower better)", "ram_plot.png", runtime_config, lower_is_better=True)
                    if any(r.get('_summary',{}).get('delta_gpu_mem_mb') is not None for r in results.values() if isinstance(r, dict)):
                        plot_paths["gpu"] = generate_ranking_plot(results, "delta_gpu_mem_mb", "Peak GPU Mem Delta Ranking (Ollama, GPU 0)", "GPU Mem Delta (MB, lower better)", "gpu_plot.png", runtime_config, lower_is_better=True)
                    else: print("[INFO] Skipping GPU plot: No GPU memory data found in results.")
                    plot_paths["accuracy_by_stage"] = generate_ranking_plot(results, "accuracy", "Accuracy by Stage", "Accuracy (%)", "accuracy_by_stage_plot.png", runtime_config, is_percentage=True, group_by_stage=True)
                    plot_paths["overall_score_by_stage"] = generate_ranking_plot(results, "overall_weighted_score", "Overall Score by Stage", "Overall Score", "overall_score_by_stage_plot.png", runtime_config, group_by_stage=True)
                except Exception as plot_err:
                     print(f"[ERROR] Failed during plot generation: {plot_err}")
                     traceback.print_exc()
            else: print("[WARN] Cannot generate plots: Visualizations enabled but Matplotlib unavailable.")
        else: print("[INFO] Skipping plot generation: Visualizations disabled.")

        try:
            print("[INFO] Generating HTML report...")
            html_content = generate_html_report(results, args.benchmark_name, plot_paths, runtime_config)
            if html_content:
                 report_path = save_report(html_content, runtime_config)
            else: print("[ERROR] HTML report generation failed (returned None).")
        except Exception as report_err:
            print(f"[ERROR] Failed generating/saving report: {report_err}")
            traceback.print_exc()

        if report_path and args.open_report: open_report_auto(report_path)
        elif report_path: print(f"[INFO] Report available at: {report_path.resolve()}")

        # --- Export Results ---
        if args.export_summary_csv:
            export_summary_csv(results, runtime_config.report_dir, args.benchmark_name)
        if args.export_details_json:
            export_details_json(results, runtime_config.report_dir, args.benchmark_name)

    else:
        print("[INFO] Skipping report generation and export: No valid results found.")
        if run_reason != "Using cache":
             print("[WARN] The benchmark run did not produce any results. Check model availability, API keys, and task definitions.")

    # --- Finalize ---
    run_duration = time.time() - run_start_time
    print(f"\nBenchmark run '{args.benchmark_name}' finished in {run_duration:.2f} seconds.")

    # Cleanup NVML if it was initialized
    if runtime_config.pynvml_available and runtime_config.pynvml:
         try:
             if hasattr(runtime_config.pynvml, 'nvmlShutdown'):
                 runtime_config.pynvml.nvmlShutdown()
         except Exception: pass

if __name__ == "__main__":
    main()