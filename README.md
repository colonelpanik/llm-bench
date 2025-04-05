# LLM Benchmark Runner

[![CI Tests](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml)

A command-line tool to benchmark local (Ollama) and remote (Google Gemini) Large Language Models (LLMs). It evaluates models against a configurable set of tasks, monitors system resources, generates a detailed HTML report with performance visualizations, and supports exporting results.

## Features

*   **Multi-Provider Support:** Benchmark models served locally via [Ollama](https://ollama.com/) (host configurable) and remotely via the Google Gemini API.
*   **Flexible Task Definition:** Define benchmark tasks in a simple JSON format (`benchmark_tasks.json`), organized by category. Includes tasks with varied prompts (e.g., different instructions, personas).
*   **Configuration File:** Manage default settings (models, paths, API keys/URLs, weights, features, timeouts) via a `config.yaml` file. CLI arguments and environment variables override file settings.
*   **Diverse Evaluation Methods:**
    *   Keyword matching (strict 'all' or flexible 'any')
    *   Weighted keyword scoring for nuanced evaluation
    *   JSON and YAML structure validation and comparison
    *   Regex-based information extraction with optional validation rules
    *   Python code execution and testing against defined test cases
    *   Classification with confidence score checking
    *   Semantic similarity comparison (optional, requires `sentence-transformers`)
*   **Resource Monitoring (Optional):**
    *   Track CPU RAM usage delta for Ollama models (requires `psutil`).
    *   Track NVIDIA GPU memory usage delta (GPU 0) for Ollama models (requires `pynvml`).
*   **Performance Metrics:** Measure API response time and tokens/second (Ollama only).
*   **Scoring:** Calculates overall accuracy, average scores for partial credit tasks, an "Ollama Performance Score", and a category-weighted "Overall Score".
*   **Reporting:**
    *   Generates a comprehensive HTML report with summary tables, performance plots (rankings for scores, accuracy, token/sec, resource usage, comparison by prompt stage), and detailed per-task results.
    *   Optional export of summary results to CSV (`--export-summary-csv`).
    *   Optional export of detailed task results to JSON (`--export-details-json`).
*   **Caching:** Caches results to speed up subsequent runs (configurable TTL).
*   **Utilities:** Includes a `--check-dependencies` flag to verify installation and basic functionality of optional libraries.
*   **Configurable:** Control models, tasks, retries, paths, optional features, scoring weights, API endpoints, and more via `config.yaml`, environment variables, and command-line arguments.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/colonelpanik/llm-bench.git
    cd llm-bench
    ```

2.  **Recommended: Create and Activate a Virtual Environment:**

    ```bash
    python -m venv .venv
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows:
    # .\.venv\Scripts\activate
    ```

3.  **Install Core Dependencies:**
    The core requirements are `requests` and `PyYAML`.

    ```bash
    pip install requests PyYAML
    ```

4.  **Install Optional Dependencies (As Needed):**
    Install libraries for features you intend to use. See `requirements.txt` for details.
    *   **RAM Monitoring (`--ram-monitor enable`):** `pip install psutil`
    *   **GPU Monitoring (`--gpu-monitor enable`):** `pip install pynvml` (Requires NVIDIA drivers/CUDA toolkit correctly installed)
    *   **Report Plots (`--visualizations enable`):** `pip install matplotlib`
    *   **Semantic Evaluation (`--semantic-eval enable`):** `pip install sentence-transformers` (Downloads model files on first use)

    You can check the status of optional dependencies using:

    ```bash
    python -m benchmark_cli --check-dependencies
    ```

## Configuration Precedence

Settings are determined in the following order (later steps override earlier ones):

1.  **Base Defaults:** Hardcoded minimal defaults in `config.py`.
2.  **`config.yaml`:** Settings loaded from the YAML configuration file (default: `config.yaml`, path configurable via `--config-file`). **This is the primary place to set your defaults.**
3.  **Environment Variables:**
    *   `GEMINI_API_KEY` overrides `api.gemini_api_key` from `config.yaml`.
    *   `OLLAMA_HOST` overrides `api.ollama_host_url` from `config.yaml` (e.g., `OLLAMA_HOST=http://some-other-ip:11434`).
4.  **Command-Line Arguments:** Any arguments provided on the command line override all previous settings (e.g., `--test-model`, `--tasks-file`, `--gemini-key`, `--ram-monitor disable`).

## Usage

The main entry point is `benchmark_cli.py`.

**Show Help:**

```bash
python -m benchmark_cli --help
```

**Basic Run (Uses defaults from `config.yaml` or base):**

```bash
python -m benchmark_cli
```

**Run specific models, overriding config defaults, clear cache:**

```bash
python -m benchmark_cli --test-model llama3:8b --test-model gemini-1.5-flash-latest --clear-cache -v
```

**Run only 'nlp' tasks category and open report:**

```bash
python -m benchmark_cli --task-set nlp --open-report
```

**Run only specific tasks by name:**

```bash
python -m benchmark_cli --task-name "Sentiment - Complex Complaint" --task-name "Code - Python Factorial"
```

**Set Gemini API Key via CLI (highest precedence):**

```bash
python -m benchmark_cli --gemini-key "YOUR_API_KEY" --test-model gemini-1.5-pro-latest
```

*(Alternatively, set `GEMINI_API_KEY` environment variable or define in `config.yaml`)*

**Use a custom configuration file:**

```bash
python -m benchmark_cli --config-file my_settings.yaml
```

**Run and export summary results to CSV:**

```bash
python -m benchmark_cli --export-summary-csv
```

**Check if optional dependencies are installed and working:**

```bash
python -m benchmark_cli --check-dependencies
```

## Configuration Files

*   **`config.yaml` (Default):** Define default models, API endpoints (`ollama_host_url`, `gemini_api_key`), paths, weights, timeouts, feature toggles, etc. See the default file for structure and comments.
*   **`benchmark_tasks.json` (Default):** Define your benchmark tasks here. Path configurable in `config.yaml` or via `--tasks-file`.
*   **`report_template.html` (Default):** Customize the HTML report template. Path configurable in `config.yaml` or via `--template-file`.

## Output

*   **HTML Report (`benchmark_report/report.html`):** Detailed report. Path configurable.
*   **Plots (`benchmark_report/images/*.png`):** PNG images embedded in the report. Path configurable.
*   **Cache Files (`benchmark_cache/cache_*.json`):** Stored results. Path configurable.
*   **Export Files (Optional):**
    *   `benchmark_report/summary_*.csv`: Summary CSV file if `--export-summary-csv` is used.
    *   `benchmark_report/details_*.json`: Detailed JSON file if `--export-details-json` is used.
*   **Console Output:** Progress, summaries, warnings, errors. Use `-v` for more detail.

## GitHub Actions / CI

This project uses GitHub Actions for automated testing. Unit tests are run automatically on pushes and pull requests to the `main` branch against multiple Python versions.

[![CI Tests](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml)

The tests mock external services (Ollama, Gemini) and do not require live instances or API keys to run in the CI environment.

## Docker Usage

A `Dockerfile` is provided for running the benchmark tool in a containerized environment with all dependencies included.

**1. Build the Docker Image:**

From the project root directory:
```bash
docker build -t llm-bench .
```

**2. Run the Benchmark:**

Running the benchmark requires connecting the container to your running Ollama instance. The method depends on your operating system.

*   **On Linux:** Use `--network=host` to share the host's network stack. Ollama running on `localhost:11434` on the host will be accessible via the same address inside the container. Mount volumes for configuration, tasks, reports, and cache.

    ```bash
    docker run --rm -it --network=host \
      -v ./config.yaml:/app/config.yaml \
      -v ./benchmark_tasks.json:/app/benchmark_tasks.json \
      -v ./benchmark_report:/app/benchmark_report \
      -v ./benchmark_cache:/app/benchmark_cache \
      llm-bench \
      --test-model llama3:8b --open-report
    ```
    *(Note: `--open-report` might not work reliably from within Docker unless you have a browser configured.)*

*   **On macOS or Windows (Docker Desktop):** `--network=host` is not typically supported. Instead, Docker provides a special DNS name `host.docker.internal` which resolves to the host machine. You need to tell `llm-bench` to use this address for Ollama.

    *   **Option A (Recommended): Using Environment Variable:** Set the `OLLAMA_HOST` environment variable when running the container.

        ```bash
        docker run --rm -it \
          -v ./config.yaml:/app/config.yaml \
          -v ./benchmark_tasks.json:/app/benchmark_tasks.json \
          -v ./benchmark_report:/app/benchmark_report \
          -v ./benchmark_cache:/app/benchmark_cache \
          -e OLLAMA_HOST="http://host.docker.internal:11434" \
          llm-bench \
          --test-model llama3:8b
        ```

    *   **Option B: Modifying `config.yaml`:** Add or modify the `api.ollama_host_url` setting in your `config.yaml` (which you mount into the container) to point to `http://host.docker.internal:11434`.

        Example `config.yaml` snippet:
        ```yaml
        api:
          # ... other keys ...
          ollama_host_url: http://host.docker.internal:11434
        ```
        Then run without the `-e OLLAMA_HOST` flag:
        ```bash
        docker run --rm -it \
          -v ./config.yaml:/app/config.yaml \
          # ... other volumes ...
          llm-bench \
          --test-model llama3:8b
        ```

**Running Specific Commands:**

You can pass any `llm-bench` command-line arguments after the image name:

```bash
# Check dependencies inside the container
docker run --rm -it llm-bench --check-dependencies

# Run specific models with verbose output
docker run --rm -it --network=host -v $(pwd):/app llm-bench --test-model mistral:7b -v
```

*(Adjust `--network=host` or add `-e OLLAMA_HOST` based on your OS for commands requiring Ollama.)*

## License

MIT License.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.