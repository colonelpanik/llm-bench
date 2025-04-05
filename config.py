import pathlib
import json
import os
import yaml # Now needed for loading config file

# --- Default Config File Path ---
DEFAULT_CONFIG_FILE = pathlib.Path('config.yaml')

# --- Default Base URLs ---
# These can be overridden by config.yaml or OLLAMA_HOST environment variable
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# Gemini base URL is less likely to change but could be made configurable if needed
DEFAULT_GEMINI_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"

# --- Base Defaults (Minimal, used if config file is missing/invalid) ---
BASE_DEFAULT_REQUEST_TIMEOUT = 180
BASE_DEFAULT_CODE_TIMEOUT = 10
BASE_DEFAULT_RETRIES = 2
BASE_DEFAULT_RETRY_DELAY = 5
BASE_DEFAULT_MODELS = ["llama3:8b"]
BASE_DEFAULT_CATEGORY_WEIGHTS = {"default": 1.0}
BASE_DEFAULT_OLLAMA_WEIGHTS = {"accuracy": 0.5, "tokens_per_sec": 0.3, "ram_efficiency": 0.2}
BASE_DEFAULT_CONFIDENCE = 0.75
BASE_PASSING_SCORE = 70.0
BASE_CACHE_TTL = 86400

# --- Function to load config from YAML file ---
def load_config_from_file(config_path):
    """Loads configuration from a YAML file."""
    if not config_path.is_file():
        print(f"[INFO] Config file '{config_path}' not found. Using base defaults and CLI args.")
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
             print(f"[WARN] Config file '{config_path}' is not a valid dictionary. Using base defaults.")
             return {}
        print(f"[INFO] Loaded configuration from '{config_path}'.")
        return config_data
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing config file '{config_path}': {e}. Using base defaults.")
        return {}
    except Exception as e:
        print(f"[ERROR] Unexpected error reading config file '{config_path}': {e}. Using base defaults.")
        return {}

# --- Runtime Configuration Holder ---
class RuntimeConfig:
    def __init__(self):
        # Module Availability Flags
        self.psutil_available = False
        self.pyyaml_available = False
        self.matplotlib_available = False
        self.sentence_transformers_available = False
        self.pynvml_available = False
        self.gpu_count = 0

        # Loaded Optional Modules
        self.psutil = None
        self.yaml = yaml
        self.plt = None
        self.mticker = None
        self.matplotlib = None
        self.SentenceTransformer = None
        self.st_util = None
        self.semantic_model = None
        self.pynvml = None

        # Operational Settings (populated from defaults, file, env, then CLI)
        self.models_to_benchmark = list(BASE_DEFAULT_MODELS)
        self.code_models_to_benchmark = []

        self.tasks_file = pathlib.Path('benchmark_tasks.json')
        self.report_dir = pathlib.Path("./benchmark_report")
        self.cache_dir = pathlib.Path("./benchmark_cache")
        self.html_template_file = pathlib.Path("./report_template.html")
        self.report_img_dir = self.report_dir / "images"
        self.report_html_file = self.report_dir / "report.html"

        self.request_timeout = BASE_DEFAULT_REQUEST_TIMEOUT
        self.code_exec_timeout = BASE_DEFAULT_CODE_TIMEOUT
        self.max_retries = BASE_DEFAULT_RETRIES
        self.retry_delay = BASE_DEFAULT_RETRY_DELAY
        self.cache_ttl = BASE_CACHE_TTL

        # API Endpoint URLs and Keys
        self.ollama_base_url = DEFAULT_OLLAMA_BASE_URL # Default, can be overridden
        self.gemini_api_url_base = DEFAULT_GEMINI_API_URL_BASE
        self.gemini_key = None # Must come from config/env/cli

        self.verbose = False

        self.ollama_score_weights = dict(BASE_DEFAULT_OLLAMA_WEIGHTS)
        self.category_weights = dict(BASE_DEFAULT_CATEGORY_WEIGHTS)
        self.default_category_weight = BASE_DEFAULT_CATEGORY_WEIGHTS.get('default', 0.5)

        self.default_min_confidence = BASE_DEFAULT_CONFIDENCE
        self.passing_score_threshold = BASE_PASSING_SCORE

        # Features enabled/disabled
        self.ram_monitor_enabled = True
        self.gpu_monitor_enabled = True
        self.visualizations_enabled = True
        self.semantic_eval_enabled = True

    def update_from_file_config(self, file_cfg):
        """Updates settings from the dictionary loaded from config.yaml."""
        if not file_cfg: return

        # API Keys & Endpoints
        api_cfg = file_cfg.get('api', {}) # Changed section name for clarity
        self.gemini_key = api_cfg.get('gemini_api_key', self.gemini_key)
        self.ollama_base_url = api_cfg.get('ollama_host_url', self.ollama_base_url)
        # Allow overriding Gemini base URL too if needed
        self.gemini_api_url_base = api_cfg.get('gemini_api_base', self.gemini_api_url_base)

        # Models
        if 'default_models' in file_cfg and isinstance(file_cfg['default_models'], list):
            self.models_to_benchmark = file_cfg['default_models']
        if 'code_models' in file_cfg and isinstance(file_cfg['code_models'], list):
            self.code_models_to_benchmark = file_cfg['code_models']

        # Paths
        paths_cfg = file_cfg.get('paths', {})
        self.tasks_file = pathlib.Path(paths_cfg.get('tasks_file', self.tasks_file))
        self.report_dir = pathlib.Path(paths_cfg.get('report_dir', self.report_dir))
        self.cache_dir = pathlib.Path(paths_cfg.get('cache_dir', self.cache_dir))
        self.html_template_file = pathlib.Path(paths_cfg.get('template_file', self.html_template_file))

        # Timeouts/Retries
        timeouts_cfg = file_cfg.get('timeouts', {})
        self.request_timeout = int(timeouts_cfg.get('request', self.request_timeout))
        self.code_exec_timeout = int(timeouts_cfg.get('code_execution_per_case', self.code_exec_timeout))
        retries_cfg = file_cfg.get('retries', {})
        self.max_retries = int(retries_cfg.get('max_retries', self.max_retries))
        self.retry_delay = int(retries_cfg.get('retry_delay', self.retry_delay))

        # Cache
        self.cache_ttl = int(file_cfg.get('cache', {}).get('ttl_seconds', self.cache_ttl))

        # Scoring Weights
        scoring_cfg = file_cfg.get('scoring', {})
        ollama_w = scoring_cfg.get('ollama_perf_score', {})
        if isinstance(ollama_w, dict): self.ollama_score_weights.update(ollama_w)
        cat_w = scoring_cfg.get('category_weights', {})
        if isinstance(cat_w, dict):
            self.category_weights = cat_w
            self.default_category_weight = cat_w.get('default', self.default_category_weight)

        # Evaluation Settings
        eval_cfg = file_cfg.get('evaluation', {})
        self.default_min_confidence = float(eval_cfg.get('default_min_confidence', self.default_min_confidence))
        self.passing_score_threshold = float(eval_cfg.get('passing_score_threshold', self.passing_score_threshold))

        # Feature Toggles
        features_cfg = file_cfg.get('features', {})
        self.ram_monitor_enabled = bool(features_cfg.get('ram_monitor', self.ram_monitor_enabled))
        self.gpu_monitor_enabled = bool(features_cfg.get('gpu_monitor', self.gpu_monitor_enabled))
        self.visualizations_enabled = bool(features_cfg.get('visualizations', self.visualizations_enabled))
        self.semantic_eval_enabled = bool(features_cfg.get('semantic_eval', self.semantic_eval_enabled))

    def update_from_environment(self):
        """Updates settings from environment variables."""
        # Gemini Key (ENV overrides File)
        env_gemini_key = os.environ.get("GEMINI_API_KEY")
        if env_gemini_key:
            self.gemini_key = env_gemini_key
            print("[INFO] Using Gemini API Key from GEMINI_API_KEY environment variable.")

        # Ollama Host (ENV overrides File)
        env_ollama_host = os.environ.get("OLLAMA_HOST")
        if env_ollama_host:
            # Basic validation: Ensure it looks like a URL prefix
            if env_ollama_host.startswith("http://") or env_ollama_host.startswith("https://"):
                # Remove trailing slash if present
                self.ollama_base_url = env_ollama_host.rstrip('/')
                print(f"[INFO] Using Ollama Host URL from OLLAMA_HOST environment variable: {self.ollama_base_url}")
            else:
                print(f"[WARN] OLLAMA_HOST environment variable ('{env_ollama_host}') does not look like a valid URL. Ignoring.")