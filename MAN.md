# NAME

llmbench - LLM Benchmark Runner

# SYNOPSIS

**llmbench** \[OPTIONS]...

# DESCRIPTION

**llmbench** is a command-line tool designed to benchmark Large Language Models (LLMs). It supports testing locally hosted models via Ollama and remote models via the Google Gemini API.

The tool runs a defined set of tasks against specified models, evaluating responses based on various criteria. It monitors system resources (CPU RAM, optionally NVIDIA GPU memory) during Ollama runs and generates a comprehensive HTML report summarizing accuracy, performance metrics, resource usage, and detailed task results.

Configuration is managed through a combination of a YAML configuration file (`config.yaml` by default), environment variables, and command-line arguments, allowing for flexible setup and execution.

# CONFIGURATION PRECEDENCE

Settings are determined in the following order, with later steps overriding earlier ones:

1.  **Base Defaults:** Minimal hardcoded defaults in the application.
2.  **YAML Configuration File:** Settings loaded from the file specified by **--config-file** (defaults to `config.yaml`). This is the primary way to set defaults for models, paths, weights, API keys, etc.
3.  **Environment Variables:** Currently, `GEMINI_API_KEY` can override the key set in the config file.
4.  **Command-Line Arguments:** Arguments provided directly on the command line take the highest precedence, overriding any previous setting.

# OPTIONS

**--config-file** *PATH*
:   Path to the YAML configuration file to load. Settings in this file provide defaults for many other options. (Default: `config.yaml`)

**--task-set** *{all,nlp,code,other}*
:   Select which category of tasks to run, based on the top-level keys in the tasks file. (Default: `all`)

**--test-model** *MODEL_NAME*
:   Specify a model name (e.g., `llama3:8b`, `gemini-1.5-flash-latest`) to include in the benchmark. Overrides the `default_models` list in the config file. Can be used multiple times.

**--gemini-key** *API_KEY*
:   API key for Google Gemini API calls. Overrides the `GEMINI_API_KEY` environment variable and the key specified in the config file. (Default: loaded from ENV or config file)

**--pull-ollama-models**
:   If specified, the tool will attempt to pull required Ollama models using the `ollama pull` API command if they are not found locally.

**--no-cache**
:   Force a fresh benchmark run, ignoring any existing cached results.

**--clear-cache**
:   Delete any existing cache file for the current benchmark name before running.

**--benchmark-name** *NAME*
:   A descriptive name for this benchmark run. Used for the report title and cache filename. (Default: "LLM Benchmark Run")

**--open-report**
:   Automatically attempt to open the generated HTML report in the default web browser after the run completes.

**--tasks-file** *PATH*
:   Path to the JSON file containing the benchmark task definitions. Overrides the `paths.tasks_file` setting in the config file. (Default: loaded from config file)

**--report-dir** *PATH*
:   Directory where the HTML report and associated images will be saved. Overrides `paths.report_dir` in the config file. (Default: loaded from config file)

**--cache-dir** *PATH*
:   Directory where benchmark result cache files will be stored. Overrides `paths.cache_dir` in the config file. (Default: loaded from config file)

**--template-file** *PATH*
:   Path to the HTML template file used for generating the report. Overrides `paths.template_file` in the config file. (Default: loaded from config file)

**--retries** *N*
:   Number of times to retry an API call on transient errors. Overrides `retries.max_retries` in the config file. (Default: loaded from config file)

**--retry-delay** *SECONDS*
:   Number of seconds to wait between retries. Overrides `retries.retry_delay` in the config file. (Default: loaded from config file)

**--category-weights** *JSON_STRING*
:   A JSON string mapping task categories to numerical weights for the "Overall Weighted Score". Overrides `scoring.category_weights` in the config file. Example: `'{"General NLP": 1.0, "Code Generation": 0.8, "default": 0.5}'`. (Default: loaded from config file)

**--ram-monitor** *{enable,disable}*
:   Enable or disable CPU RAM usage monitoring. Overrides `features.ram_monitor` in the config file. Requires `psutil`. (Default: loaded from config file)

**--gpu-monitor** *{enable,disable}*
:   Enable or disable NVIDIA GPU memory usage monitoring. Overrides `features.gpu_monitor` in the config file. Requires `pynvml`. (Default: loaded from config file)

**--visualizations** *{enable,disable}*
:   Enable or disable plot generation in the HTML report. Overrides `features.visualizations` in the config file. Requires `matplotlib`. (Default: loaded from config file)

**--semantic-eval** *{enable,disable}*
:   Enable or disable semantic similarity evaluation. Overrides `features.semantic_eval` in the config file. Requires `sentence-transformers`. (Default: loaded from config file)

**-v**, **--verbose**
:   Enable verbose logging.

**-h**, **--help**
:   Show the help message and exit.

# EXAMPLES

**Run benchmark with defaults from config.yaml:**
```bash
llmbench
```

**Run with a specific config file:**
```bash
llmbench --config-file production_settings.yaml
```

**Run specific models, overriding config and clearing cache:**
```bash
llmbench --test-model llama3:8b --test-model mistral:7b --clear-cache
```

**Run only code tasks, overriding the tasks file path:**
```bash
llmbench --task-set code --tasks-file custom_code_tasks.json --open-report
```

# FILES

**config.yaml (Default)**
:   The primary YAML configuration file. Path configurable via **--config-file**. Contains defaults for models, paths, API keys, weights, features, etc.

**benchmark_tasks.json (Default, configurable)**
:   JSON file defining benchmark tasks. Path configured in `config.yaml` or via **--tasks-file**.

**report_template.html (Default, configurable)**
:   HTML template for report generation. Path configured in `config.yaml` or via **--template-file**.

**benchmark_report/report.html (Default, configurable)**
:   Generated HTML report file. Location configured in `config.yaml` or via **--report-dir**.

**benchmark_report/images/*.png (Default, configurable)**
:   Image files for plots. Location based on report directory.

**benchmark_cache/cache_*.json (Default, configurable)**
:   Cache files storing results. Location configured in `config.yaml` or via **--cache-dir**.

# ENVIRONMENT VARIABLES

**GEMINI_API_KEY**
:   Overrides the Gemini API key specified in the config file. The **--gemini-key** command-line option takes highest precedence.

# BUGS

Report bugs via GitHub issues for the project.

# AUTHOR

[Your Name or Organization]