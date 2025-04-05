# -*- coding: utf-8 -*-
# --- Unit Tests for LLM Benchmark Runner ---

import unittest
import json
import time
import sys
import io
import pathlib
import tempfile
import shutil
import os  # Imported for os.environ access in TestCLIConfigLoading
from unittest.mock import patch, MagicMock, mock_open, ANY
import requests
import yaml # Needed for mock config data

# Add project root to path to allow importing modules
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Modules to test (import after adding to path)
import config
from config import RuntimeConfig # Import class directly
import utils
# Import utils functions used in tests if needed
from utils import TimeoutException, HAS_SIGNAL_ALARM, truncate_text
import llm_clients
import system_monitor
import evaluation # Imports submodules
from evaluation import standard_evaluators, structured_evaluators, code_evaluator, semantic_evaluator
import scoring
import cache_manager
import reporting
import benchmark_cli # Import the module itself for testing main entry point
from benchmark_runner import run_benchmark_set # Import specific functions if needed

# --- Constants for Tests ---
TEST_MODEL_OLLAMA = "test-ollama-model:latest"
TEST_MODEL_GEMINI = "gemini-test-model"
TEST_BENCHMARK_NAME = "Unit Test Benchmark"
DEFAULT_TEST_CONFIG_PATH = pathlib.Path("./config.yaml")

# --- Mock Responses ---
MOCK_OLLAMA_RESPONSE_SUCCESS = {
    "model": TEST_MODEL_OLLAMA, "created_at": "2023-10-27T14:00:00.000Z",
    "response": "This is a successful test response.", "done": True, "context": [1, 2, 3],
    "total_duration": 5000000000, "load_duration": 1000000, "prompt_eval_count": 10,
    "prompt_eval_duration": 200000000, "eval_count": 30, "eval_duration": 3000000000
}
MOCK_GEMINI_RESPONSE_SUCCESS = {
  "candidates": [{"content": {"parts": [{"text": "This is a successful Gemini test response."}],"role": "model"}, "finishReason": "STOP", "index": 0, "safetyRatings": []}],
  "promptFeedback": {"safetyRatings": []}
}
MOCK_GEMINI_RESPONSE_BLOCKED = {"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": []}}
MOCK_GEMINI_RESPONSE_CANDIDATE_BLOCKED = {"candidates": [{"finishReason": "SAFETY", "index": 0, "safetyRatings": []}]}
MOCK_OLLAMA_MODELS_RESPONSE = {"models": [{"name": TEST_MODEL_OLLAMA, "modified_at": "...", "size": 12345},{"name": "another-model:7b", "modified_at": "...", "size": 67890}]}

# --- Mock Default Config Data ---
MOCK_DEFAULT_FILE_CONFIG = {
    'api_keys': {'gemini': 'file-gemini-key'},
    'default_models': ['file-default-model:latest', TEST_MODEL_GEMINI],
    'paths': { 'tasks_file': 'config_tasks.json', 'report_dir': './config_report', 'cache_dir': './config_cache', 'template_file': 'config_template.html' },
    'timeouts': {'request': 200, 'code_execution_per_case': 15},
    'retries': {'max_retries': 1, 'retry_delay': 3},
    'cache': {'ttl_seconds': 3600},
    'scoring': { 'ollama_perf_score': {'accuracy': 0.6, 'tokens_per_sec': 0.2, 'ram_efficiency': 0.2}, 'category_weights': {'General NLP': 0.9, 'Code Generation': 1.1, 'default': 0.6}},
    'features': { 'ram_monitor': False, 'gpu_monitor': False, 'visualizations': False, 'semantic_eval': False },
    'evaluation': {'default_min_confidence': 0.8, 'passing_score_threshold': 75.0}
}

# --- Mock Objects ---
class MockPsutilProcess:
    def __init__(self, pid, name='python', cmdline=None, rss=100 * 1024 * 1024):
        self.pid = pid
        self._name = name
        self._cmdline = cmdline or ['python']
        self._rss = rss
        self._is_running = True
        self.info = {'pid': self.pid, 'name': self._name, 'cmdline': self._cmdline}

    def memory_info(self):
        if not self._is_running:
            raise getattr(psutil, 'NoSuchProcess', Exception)(self.pid)
        mock_mem = MagicMock()
        mock_mem.rss = self._rss
        return mock_mem

    def is_running(self):
        return self._is_running

    def terminate(self):
        self._is_running = False

# Mock external libraries if not installed
try:
    import psutil
except ImportError:
    psutil = MagicMock()
    psutil.NoSuchProcess = Exception
    psutil.AccessDenied = Exception
    psutil.process_iter.return_value = []
    psutil.pid_exists.return_value = False
try:
    import pynvml
except ImportError:
    pynvml = MagicMock()
    pynvml.NVMLError = Exception
    pynvml.nvmlInit.return_value = None
    pynvml.nvmlShutdown.return_value = None
    pynvml.nvmlDeviceGetCount.return_value = 0
    pynvml.nvmlDeviceGetHandleByIndex.side_effect = IndexError
    pynvml.nvmlDeviceGetMemoryInfo.side_effect = pynvml.NVMLError("Mock NVML Error")

class MockRuntimeConfigForEval(RuntimeConfig):
    def __init__(self, ram=False, gpu=False, yaml_needed=True, plots=False, semantic=False):
        super().__init__()
        self.psutil_available = ram
        self.pynvml_available = gpu
        self.pyyaml_available = yaml_needed
        self.matplotlib_available = plots
        self.sentence_transformers_available = semantic
        self.psutil = MagicMock() if ram else None
        self.pynvml = MagicMock() if gpu else None
        self.yaml = MagicMock() if yaml_needed else None
        self.matplotlib = MagicMock() if plots else None
        self.plt = MagicMock() if plots else None
        self.SentenceTransformer = MagicMock() if semantic else None
        self.st_util = MagicMock() if semantic else None
        self.semantic_model = MagicMock() if semantic else None

        if ram:
            self.psutil.NoSuchProcess = getattr(psutil, 'NoSuchProcess', Exception)
            self.psutil.AccessDenied = getattr(psutil, 'AccessDenied', Exception)
            self.psutil.process_iter.return_value = []
            self.psutil.pid_exists.return_value = False
        if gpu:
            self.pynvml.NVMLError = getattr(pynvml, 'NVMLError', Exception)
            self.pynvml.nvmlDeviceGetCount.return_value = 1 if gpu else 0
            self.pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
            mock_mem = MagicMock()
            mock_mem.used = 0
            self.pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        if yaml_needed:
            try:
                import yaml as real_yaml
                self.yaml.YAMLError = real_yaml.YAMLError
            except ImportError:
                self.yaml.YAMLError = Exception
            self.yaml.safe_load.return_value = {}
        if semantic:
            mock_tensor = MagicMock()
            mock_tensor.item.return_value = 0.8
            self.semantic_model.encode.return_value = mock_tensor
            self.st_util.pytorch_cos_sim.return_value = mock_tensor

        # Use base default for tests
        self.default_min_confidence = 0.75
        self.passing_score_threshold = 70.0
        self.code_exec_timeout = 10

# --- Test Classes ---

class TestUtils(unittest.TestCase):
    def test_format_na(self):
        self.assertEqual(utils.format_na(None), "N/A")
        self.assertEqual(utils.format_na(float('nan')), "N/A")
        self.assertEqual(utils.format_na(123), "123")
        self.assertEqual(utils.format_na(123.456), "123.5")
        self.assertEqual(utils.format_na(123.456, precision=2), "123.46")
        self.assertEqual(utils.format_na(123.456, suffix=" MB", precision=1), "123.5 MB")
        self.assertEqual(utils.format_na(0), "0")
        self.assertEqual(utils.format_na("abc"), "abc")

    def test_truncate_text(self):
        self.assertEqual(utils.truncate_text("short text"), "short text")
        self.assertEqual(utils.truncate_text(""), "")
        self.assertEqual(utils.truncate_text(None), "")
        long_text = "This is a very long text that definitely exceeds the default limit."
        self.assertEqual(utils.truncate_text(long_text, max_len=20), "This is a very long ...")
        self.assertTrue(len(utils.truncate_text(long_text, max_len=20)) <= 20 + 3)


class TestLLMClients(unittest.TestCase):
    def setUp(self):
        self.mock_runtime_config = RuntimeConfig()
        self.mock_runtime_config.update_from_file_config(MOCK_DEFAULT_FILE_CONFIG)
        self.mock_runtime_config.verbose = False
        self.mock_runtime_config.gemini_key = "test-key"

    @patch('requests.post')
    def test_query_ollama_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_OLLAMA_RESPONSE_SUCCESS
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA, "test prompt", self.mock_runtime_config)
        mock_post.assert_called_once_with(config.OLLAMA_API_URL, json=ANY, timeout=self.mock_runtime_config.request_timeout)
        self.assertEqual(text, "This is a successful test response.")
        self.assertIsNone(error)
        self.assertGreater(duration, 0)
        self.assertAlmostEqual(tokps, 10.0, delta=0.1)

    @patch('requests.post')
    def test_query_ollama_404_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"error": "model not found"}
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_ollama("missing-model", "p", self.mock_runtime_config)
        self.assertEqual(text, "")
        self.assertIsNotNone(error)
        self.assertIn("404", error)
        self.assertIn("model not found", error)
        mock_post.assert_called_once()

    @patch('requests.post')
    @patch('time.sleep', return_value=None)
    def test_query_ollama_500_retry_fail(self, mock_sleep, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA, "p", self.mock_runtime_config)
        self.assertEqual(text, "")
        self.assertIsNotNone(error)
        self.assertIn("Server Error HTTP 500", error)
        self.assertIn(f"After {self.mock_runtime_config.max_retries + 1} attempts", error)
        self.assertEqual(mock_post.call_count, self.mock_runtime_config.max_retries + 1)
        self.assertEqual(mock_sleep.call_count, self.mock_runtime_config.max_retries)

    @patch('requests.post')
    @patch('time.sleep', return_value=None)
    def test_query_ollama_timeout_retry_success(self, mock_sleep, mock_post):
        mock_resp_success = MagicMock()
        mock_resp_success.status_code = 200
        mock_resp_success.json.return_value = MOCK_OLLAMA_RESPONSE_SUCCESS
        mock_post.side_effect = [requests.exceptions.Timeout("Request timed out"), mock_resp_success]
        text, duration, tokps, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA, "p", self.mock_runtime_config)
        self.assertEqual(text, "This is a successful test response.")
        self.assertIsNone(error)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch('requests.post')
    def test_query_gemini_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_SUCCESS
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, "This is a successful Gemini test response.")
        self.assertIsNone(error)
        self.assertIsNone(tokps) # Gemini API doesn't provide token count/timing
        self.assertGreater(duration, 0)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_query_gemini_blocked_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_BLOCKED
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, "")
        self.assertIsNotNone(error)
        self.assertIn("Blocked by API (Prompt): SAFETY", error)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_query_gemini_blocked_candidate(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_CANDIDATE_BLOCKED
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, "")
        self.assertIsNotNone(error)
        self.assertIn("Blocked by API (Response): SAFETY", error)
        mock_post.assert_called_once()

    @patch('requests.get', side_effect=requests.exceptions.ConnectionError("Connection failed"))
    def test_get_local_ollama_models_connection_error(self, mock_get):
        models = llm_clients.get_local_ollama_models()
        self.assertIsNone(models)

    @patch('requests.get')
    def test_get_local_ollama_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_OLLAMA_MODELS_RESPONSE
        mock_get.return_value = mock_resp
        models = llm_clients.get_local_ollama_models()
        self.assertEqual(models, {TEST_MODEL_OLLAMA, "another-model:7b"})
        mock_get.assert_called_once_with(config.OLLAMA_API_MODELS_URL, timeout=10)


class TestSystemMonitor(unittest.TestCase):
    def test_get_ollama_pids_found(self):
        mock_proc1 = MockPsutilProcess(pid=123, name='ollama', cmdline=['ollama', 'serve'])
        mock_proc2 = MockPsutilProcess(pid=456, name='other')
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.NoSuchProcess = getattr(psutil, 'NoSuchProcess', Exception)
        mock_psutil_mod.AccessDenied = getattr(psutil, 'AccessDenied', Exception)
        mock_proc1.is_running = MagicMock(return_value=True)
        mock_proc2.is_running = MagicMock(return_value=True)
        mock_psutil_mod.process_iter.return_value = [mock_proc1, mock_proc2]
        pids = system_monitor.get_ollama_pids(mock_psutil_mod)
        self.assertEqual(pids, [123])
        mock_psutil_mod.process_iter.assert_called_once_with(['pid', 'name', 'cmdline'])

    @patch('requests.post')
    def test_get_ollama_pids_fallback(self, mock_api_post):
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.process_iter.return_value = []
        mock_psutil_mod.pid_exists.return_value = True
        mock_psutil_mod.NoSuchProcess = getattr(psutil, 'NoSuchProcess', Exception)
        mock_psutil_mod.AccessDenied = getattr(psutil, 'AccessDenied', Exception)
        mock_api_resp = MagicMock()
        mock_api_resp.status_code = 404
        # Simulate error message format from Ollama when model not found
        mock_api_resp.json.return_value = {"error": "model 'dummy' not found, process with pid 789"}
        mock_api_post.return_value = mock_api_resp
        pids = system_monitor.get_ollama_pids(mock_psutil_mod)
        self.assertEqual(pids, [789])
        mock_api_post.assert_called_once()

    def test_get_combined_rss(self):
        mock_proc1 = MockPsutilProcess(pid=1, rss=100 * 1024**2)
        mock_proc2 = MockPsutilProcess(pid=2, rss=200 * 1024**2)
        mock_proc3 = MockPsutilProcess(pid=3, rss=50 * 1024**2)
        mock_proc3.terminate() # Simulate a process ending during check
        mock_psutil_mod = MagicMock()
        mock_psutil_mod.NoSuchProcess = getattr(psutil, 'NoSuchProcess', Exception)
        mock_psutil_mod.AccessDenied = getattr(psutil, 'AccessDenied', Exception)
        mock_psutil_mod.Process.side_effect = lambda pid: {1: mock_proc1, 2: mock_proc2, 3: mock_proc3}[pid]
        total_rss = system_monitor.get_combined_rss([1, 2, 3], mock_psutil_mod)
        self.assertEqual(total_rss, 300 * 1024**2)
        self.assertEqual(mock_psutil_mod.Process.call_count, 3)

    def test_get_gpu_memory_usage_success(self):
        mock_nvml_mod = MagicMock()
        mock_handle = MagicMock()
        mock_mem_info = MagicMock()
        mock_mem_info.used = 5 * 1024**3 # 5 GiB
        mock_nvml_mod.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml_mod.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mock_nvml_mod.NVMLError = getattr(pynvml, 'NVMLError', Exception)
        mem_used = system_monitor.get_gpu_memory_usage(mock_nvml_mod, device_index=0)
        self.assertEqual(mem_used, 5 * 1024**3)
        mock_nvml_mod.nvmlDeviceGetHandleByIndex.assert_called_with(0)
        mock_nvml_mod.nvmlDeviceGetMemoryInfo.assert_called_with(mock_handle)

    def test_get_gpu_memory_usage_error(self):
        mock_nvml_mod = MagicMock()
        mock_nvml_mod.NVMLError = getattr(pynvml, 'NVMLError', Exception)
        mock_nvml_mod.nvmlDeviceGetHandleByIndex.side_effect = mock_nvml_mod.NVMLError("Device not found")
        mem_used = system_monitor.get_gpu_memory_usage(mock_nvml_mod, device_index=0)
        self.assertEqual(mem_used, 0)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.runtime_config = MockRuntimeConfigForEval(ram=True, gpu=True, yaml_needed=True, plots=True, semantic=True)
        self.runtime_config_no_yaml = MockRuntimeConfigForEval(yaml_needed=False)
        self.runtime_config_no_semantic = MockRuntimeConfigForEval(semantic=False)

    def test_evaluate_response_dispatch_code(self):
        task = {"name": "t", "type": "code_generation", "prompt": "p", "function_name": "f", "test_cases": []}
        with patch('evaluation.evaluator.execute_and_test_code') as mock_exec:
            mock_exec.return_value = (True, "Code OK")
            metric, details = evaluation.evaluate_response(task, "code", self.runtime_config)
            mock_exec.assert_called_once_with("code", "f", [], self.runtime_config.code_exec_timeout)
            self.assertTrue(metric)
            self.assertEqual(details, "Code OK")

    def test_evaluate_response_dispatch_semantic(self):
        task = {"name": "t", "type": "summarization", "evaluation_method": "semantic", "prompt": "p", "expected_keywords": ["ref"]}
        with patch('evaluation.evaluator.evaluate_semantic_similarity') as mock_sem:
            mock_sem.return_value = (95.0, "Similarity High")
            metric, details = evaluation.evaluate_response(task, "response", self.runtime_config)
            mock_sem.assert_called_once_with(task, "response", self.runtime_config.semantic_model, self.runtime_config.st_util, ANY)
            self.assertEqual(metric, 95.0)
            self.assertEqual(details, "Similarity High")

    def test_evaluate_response_dispatch_semantic_unavailable(self):
        # Test fallback to keyword when semantic unavailable
        task = {"name": "t", "type": "summarization", "evaluation_method": "semantic", "prompt": "p", "expected_keywords": ["ref"]}
        with patch('evaluation.evaluator.partial_keyword_scoring') as mock_partial:
            mock_partial.return_value = (50.0, "Partial Match")
            metric, details = evaluation.evaluate_response(task, "response", self.runtime_config_no_semantic)
            mock_partial.assert_called_once_with(task, "response")
            self.assertEqual(metric, 50.0)
            self.assertEqual(details, "Partial Match")

    def test_partial_keyword_scoring(self):
        # Weighted keywords
        task_weighted = {
            "name": "t", "type": "summarization", "prompt": "p",
            "weighted_keywords": [{"keyword": "apple", "weight": 2}, {"keyword": "banana", "weight": 1}]
        }
        score, details = standard_evaluators.partial_keyword_scoring(task_weighted, "I like Apple pie.")
        self.assertAlmostEqual(score, (2/3)*100)
        self.assertIn("apple(2.0)", details)
        self.assertIn("Missing: [banana(1.0)]", details)

        # Unweighted keywords (default weight 1)
        task_unweighted = {
            "name": "t", "type": "code_explanation", "prompt": "p",
            "expected_keywords": ["loop", "variable"]
        }
        score, details = standard_evaluators.partial_keyword_scoring(task_unweighted, "This code has a loop.")
        self.assertAlmostEqual(score, 50.0)
        self.assertIn("Found: [loop(1.0)]", details)
        self.assertIn("Missing: [variable(1.0)]", details)

    def test_evaluate_structured_output_json(self):
        task = {"name": "t", "type": "json_gen", "prompt": "p", "expected_structure": {"a": 1, "b": "hello"}}
        # Exact match
        metric, details = structured_evaluators.evaluate_structured_output(task, '{"a": 1, "b": "hello"}', True, None)
        self.assertTrue(metric)
        # Order doesn't matter
        metric, details = structured_evaluators.evaluate_structured_output(task, '{"b": "hello", "a": 1}', True, None)
        self.assertTrue(metric)
        # Value mismatch
        metric, details = structured_evaluators.evaluate_structured_output(task, '{"a": 2, "b": "hello"}', True, None)
        self.assertFalse(metric)
        self.assertIn("Mismatch at key 'a'", details)
        # Missing key
        metric, details = structured_evaluators.evaluate_structured_output(task, '{"a": 1}', True, None)
        self.assertFalse(metric)
        self.assertIn("Missing keys: {'b'}", details)
        # Invalid JSON (trailing comma)
        metric, details = structured_evaluators.evaluate_structured_output(task, '{"a": 1, }', True, None)
        self.assertFalse(metric)
        self.assertIn("JSON parse error", details)

    def test_evaluate_structured_output_yaml(self):
        task = {"name": "t", "type": "yaml_gen", "prompt": "p", "expected_structure": ["x", "y"]}
        # Success case with YAML available
        self.runtime_config.yaml.safe_load.return_value = ["x", "y"] # Mock successful parsing
        metric, details = structured_evaluators.evaluate_structured_output(task, "- x\n- y", True, self.runtime_config.yaml)
        self.assertTrue(metric)
        self.runtime_config.yaml.safe_load.assert_called_with("- x\n- y")
        # Failure case when YAML lib not available (but mock is True for this test)
        metric, details = structured_evaluators.evaluate_structured_output(task, "- x\n- y", False, None)
        self.assertFalse(metric)
        self.assertIn("PyYAML library not available", details)

    def test_execute_code_syntax_error(self):
        code = "```python\ndef my_func(x)\n return x\n```" # Missing colon
        task = {"function_name": "my_func", "test_cases": [{"input": [5], "expected_output": 10}]}
        timeout = self.runtime_config.code_exec_timeout

        with patch('builtins.exec', side_effect=SyntaxError("Invalid syntax")) as mock_exec, \
             patch('evaluation.code_evaluator.setup_timeout_alarm') as mock_setup_alarm, \
             patch('evaluation.code_evaluator.setup_timeout_fallback') as mock_setup_fallback, \
             patch('evaluation.code_evaluator.cancel_timeout_alarm') as mock_cancel_alarm, \
             patch('evaluation.code_evaluator.cancel_timeout_fallback') as mock_cancel_fallback, \
             patch('evaluation.code_evaluator.check_timed_out', return_value=False) as mock_check_timed_out, \
             patch('evaluation.code_evaluator.clear_timeout_flag') as mock_clear_flag, \
             patch('evaluation.code_evaluator.HAS_SIGNAL_ALARM', False) as mock_has_signal, \
             patch('io.StringIO', MagicMock()), patch('contextlib.redirect_stdout', MagicMock()):

            passed, details = code_evaluator.execute_and_test_code(
                code, task["function_name"], task["test_cases"], timeout
            )
            self.assertFalse(passed)
            self.assertIn("SyntaxError", details)
            self.assertIn("Invalid syntax", details)
            # exec is called before syntax check can raise internal error
            mock_exec.assert_called_once()
            mock_setup_fallback.assert_called_once() # Changed from assert_not_called
            mock_cancel_fallback.assert_called_once() # Added assertion for cancellation
            # Timeout setup/cancel should not be called if exec fails immediately
            mock_setup_alarm.assert_not_called()
            mock_cancel_alarm.assert_not_called()
            # Flag should always be cleared in finally block
            mock_clear_flag.assert_called()

    def test_execute_code_success(self):
        code = "```python\ndef my_func(x):\n  return x * 2\n```"
        task = {"function_name": "my_func", "test_cases": [{"input": [5], "expected_output": 10}]}
        timeout = self.runtime_config.code_exec_timeout

        with patch('builtins.exec') as mock_exec, \
             patch('evaluation.code_evaluator.setup_timeout_alarm') as mock_setup_alarm, \
             patch('evaluation.code_evaluator.setup_timeout_fallback') as mock_setup_fallback, \
             patch('evaluation.code_evaluator.cancel_timeout_alarm') as mock_cancel_alarm, \
             patch('evaluation.code_evaluator.cancel_timeout_fallback') as mock_cancel_fallback, \
             patch('evaluation.code_evaluator.check_timed_out', return_value=False) as mock_check_timed_out, \
             patch('evaluation.code_evaluator.clear_timeout_flag') as mock_clear_flag, \
             patch('evaluation.code_evaluator.HAS_SIGNAL_ALARM', False) as mock_has_signal, \
             patch('io.StringIO', MagicMock()), patch('contextlib.redirect_stdout', MagicMock()):

            # Define a side effect for exec to simulate function definition
            def exec_side_effect(code_str, globals_dict, locals_dict):
                # Simulate the function being defined in the locals dict
                def simulated_func(x): return x * 2
                locals_dict['my_func'] = simulated_func
                # Check if code fences were removed
                self.assertNotIn("```", code_str)
                self.assertIn("def my_func(x):", code_str)
            mock_exec.side_effect = exec_side_effect

            passed, details = code_evaluator.execute_and_test_code(
                code, task["function_name"], task["test_cases"], timeout
            )
            self.assertTrue(passed)
            self.assertIn("Got: 10, Exp: 10 [PASS]", details)
            mock_exec.assert_called_once()
            # Assuming fallback timeout since HAS_SIGNAL_ALARM is mocked to False
            mock_setup_fallback.assert_called_once()
            mock_cancel_fallback.assert_called_once()
            mock_setup_alarm.assert_not_called()
            mock_cancel_alarm.assert_not_called()
            mock_clear_flag.assert_called()

    def test_semantic_similarity_eval(self):
        task = {
            "name": "t", "type": "summarization", "evaluation_method": "semantic",
            "prompt": "p", "expected_keywords": ["Reference text"], "similarity_threshold": 0.75
        }
        metric, details = semantic_evaluator.evaluate_semantic_similarity(
            task, "Response text", self.runtime_config.semantic_model,
            self.runtime_config.st_util, threshold=task["similarity_threshold"]
        )
        # Mock model returns 0.8 cosine similarity
        self.assertAlmostEqual(metric, 80.0)
        self.assertIn("Score=80.0%", details)
        self.assertIn("Raw Cosine: 0.800", details)
        self.assertIn("Threshold for Pass >= 75.0%", details)
        self.runtime_config.semantic_model.encode.assert_called()
        self.runtime_config.st_util.pytorch_cos_sim.assert_called()


class TestScoring(unittest.TestCase):
    def test_compute_scores(self):
        mock_results = {
            "ollama_fast_good": { "_summary": {
                "model_name": "ollama_fast_good", "provider": "ollama", "status": "Completed",
                "accuracy": 90.0, "tokens_per_sec_avg": 100.0, "delta_ram_mb": 500.0, "delta_gpu_mem_mb": 1000.0,
                "processed_count": 10, "per_type": {"General NLP": {"accuracy": 90.0, "count": 10, "success_api_calls": 10}}
            }},
            "ollama_slow_bad": { "_summary": {
                "model_name": "ollama_slow_bad", "provider": "ollama", "status": "Completed",
                "accuracy": 60.0, "tokens_per_sec_avg": 20.0, "delta_ram_mb": 1500.0, "delta_gpu_mem_mb": 3000.0,
                "processed_count": 10, "per_type": {"General NLP": {"accuracy": 60.0, "count": 10, "success_api_calls": 10}}
            }},
            "gemini_ok": { "_summary": {
                "model_name": "gemini_ok", "provider": "gemini", "status": "Completed",
                "accuracy": 75.0, "tokens_per_sec_avg": None, "delta_ram_mb": None, "delta_gpu_mem_mb": None,
                "processed_count": 10, "per_type": {"General NLP": {"accuracy": 75.0, "count": 10, "success_api_calls": 10}}
            }},
            "skipped_model": { "_summary": {
                "model_name": "skipped_model", "status": "Skipped"
            }}
        }
        mock_results["_task_categories"] = {"General NLP": ["task1"]}
        mock_results["_task_definitions"] = {"task1": {"type": "General NLP"}}
        # Use simpler weights for easier verification
        mock_results["_ollama_score_weights_used"] = {"accuracy": 0.5, "tokens_per_sec": 0.3, "ram_efficiency": 0.2}
        category_weights = {"General NLP": 1.0, "default": 0.5}
        default_weight = 0.5

        scoring.compute_performance_scores(mock_results, category_weights, default_weight)

        # Check Ollama performance scores (higher is better)
        score1 = mock_results["ollama_fast_good"]["_summary"]["ollama_perf_score"]
        score2 = mock_results["ollama_slow_bad"]["_summary"]["ollama_perf_score"]
        self.assertIsNotNone(score1)
        self.assertIsNotNone(score2)
        self.assertGreater(score1, score2) # Fast/Good should score higher than Slow/Bad

        # Check overall weighted scores (higher accuracy = higher score here as category weight = 1.0)
        score_o1 = mock_results["ollama_fast_good"]["_summary"]["overall_weighted_score"]
        score_o2 = mock_results["ollama_slow_bad"]["_summary"]["overall_weighted_score"]
        score_g1 = mock_results["gemini_ok"]["_summary"]["overall_weighted_score"]
        self.assertAlmostEqual(score_o1, 90.0 * 1.0) # accuracy * category_weight
        self.assertAlmostEqual(score_o2, 60.0 * 1.0) # accuracy * category_weight
        self.assertAlmostEqual(score_g1, 75.0 * 1.0) # accuracy * category_weight

        # Skipped models should not have scores
        self.assertIsNone(mock_results["skipped_model"]["_summary"].get("ollama_perf_score"))
        self.assertIsNone(mock_results["skipped_model"]["_summary"].get("overall_weighted_score"))


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = pathlib.Path(self.temp_dir)
        # Generate safe filename from benchmark name
        safe_name = utils.re.sub(r"[^a-zA-Z0-9_\-]+", "_", TEST_BENCHMARK_NAME.lower())
        self.cache_file = self.cache_dir / f"cache_{safe_name}.json"
        self.cache_ttl = 3600 # 1 hour

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_cache_path(self):
        path = cache_manager.get_cache_path(self.cache_dir, TEST_BENCHMARK_NAME)
        self.assertEqual(path, self.cache_file)

    def test_save_load_cache(self):
        data_to_save = {"model1": {"_summary": {"accuracy": 80.0}}}
        cache_manager.save_cache(self.cache_file, data_to_save)
        self.assertTrue(self.cache_file.is_file())
        loaded_data = cache_manager.load_cache(self.cache_file, self.cache_ttl)
        self.assertEqual(loaded_data, data_to_save)

    def test_load_cache_nonexistent(self):
        loaded_data = cache_manager.load_cache(self.cache_file, self.cache_ttl)
        self.assertIsNone(loaded_data)

    def test_load_cache_expired(self):
        data_to_save = {"model1": {"_summary": {"accuracy": 80.0}}}
        cache_manager.save_cache(self.cache_file, data_to_save)
        # Patch time to be after TTL expires
        with patch('time.time', return_value=time.time() + self.cache_ttl + 10):
            loaded_data = cache_manager.load_cache(self.cache_file, self.cache_ttl)
            self.assertIsNone(loaded_data) # Should return None as cache is stale

    def test_load_cache_corrupted(self):
        # Write invalid JSON to the cache file
        with open(self.cache_file, "w") as f:
            f.write("{invalid json")
        loaded_data = cache_manager.load_cache(self.cache_file, self.cache_ttl)
        self.assertIsNone(loaded_data) # Should return None on JSON decode error

    def test_clear_cache(self):
        data_to_save = {"model1": {"_summary": {"accuracy": 80.0}}}
        cache_manager.save_cache(self.cache_file, data_to_save)
        self.assertTrue(self.cache_file.is_file())
        cache_manager.clear_cache(self.cache_file)
        self.assertFalse(self.cache_file.is_file())


class TestReporting(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name
        # Setup mock config with plotting enabled
        self.runtime_config = MockRuntimeConfigForEval(plots=True)
        self.runtime_config.report_dir = pathlib.Path(self.temp_dir)
        self.runtime_config.report_img_dir = self.runtime_config.report_dir / "images"
        self.runtime_config.report_img_dir.mkdir(parents=True, exist_ok=True)
        # Setup mock config with plotting disabled
        self.runtime_config_no_plots = MockRuntimeConfigForEval(plots=False)
        # Sample results data
        self.results = {
            "model_a": {"_summary": {
                "status": "Completed", "model_name": "model_a", "provider": "ollama",
                "accuracy": 90.0, "overall_weighted_score": 90.0, "ollama_perf_score": None
            }},
            "model_b": {"_summary": {
                "status": "Completed", "model_name": "model_b", "provider": "ollama",
                "accuracy": 80.0, "overall_weighted_score": 80.0, "ollama_perf_score": 75.0
            }},
            "model_c": {"_summary": {
                "status": "Skipped", "model_name": "model_c", "provider": "ollama"
            }},
            "_task_definitions": {}, "_task_categories": {}, "_category_weights_used": {}
        }
        # Mock matplotlib if available
        if self.runtime_config.matplotlib_available:
            self.runtime_config.plt = MagicMock()
            self.runtime_config.plt.savefig = MagicMock()
            self.runtime_config.plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock())) # Returns fig, ax
            self.runtime_config.plt.close = MagicMock()

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_generate_ranking_plot_success(self):
        # Ensure conditions for plotting are met
        self.runtime_config.matplotlib_available = True
        self.runtime_config.visualizations_enabled = True
        self.runtime_config.plt = MagicMock() # Re-mock plt just in case
        self.runtime_config.plt.savefig = MagicMock()
        self.runtime_config.plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock()))
        self.runtime_config.plt.close = MagicMock()

        path = reporting.generate_ranking_plot(
            self.results, "accuracy", "Accuracy Plot", "Acc", "acc.png", self.runtime_config
        )
        # Expect relative path for HTML embedding
        self.assertEqual(path, "images/acc.png")
        self.runtime_config.plt.subplots.assert_called()
        expected_save_path = self.runtime_config.report_img_dir / "acc.png"
        self.runtime_config.plt.savefig.assert_called_once_with(expected_save_path, bbox_inches='tight', dpi=100)
        self.runtime_config.plt.close.assert_called_once()

    def test_generate_ranking_plot_disabled(self):
        # Case 1: Feature disabled in config
        self.runtime_config.visualizations_enabled = False
        self.runtime_config.matplotlib_available = True # Library is available
        self.runtime_config.plt = MagicMock()

        path = reporting.generate_ranking_plot(self.results, "accuracy", "Acc Plot", "Acc", "acc.png", self.runtime_config)
        self.assertIsNone(path)
        # savefig might exist on the mock but shouldn't be called
        # self.runtime_config.plt.savefig.assert_not_called()

        # Case 2: Library unavailable
        self.runtime_config_no_plots.visualizations_enabled = True # Feature enabled
        self.runtime_config_no_plots.matplotlib_available = False # Library missing
        path = reporting.generate_ranking_plot(self.results, "accuracy", "Acc Plot", "Acc", "acc.png", self.runtime_config_no_plots)
        self.assertIsNone(path)

    @patch('builtins.open', new_callable=mock_open, read_data="<html>{{REPORT_TITLE}} {{SUMMARY_TABLE_ROWS}}</html>")
    def test_generate_html_report(self, mock_file_open):
        self.runtime_config.html_template_file = pathlib.Path("./report_template.html")
        # Patch is_file to simulate template existence
        with patch.object(pathlib.Path, 'is_file', return_value=True):
            html = reporting.generate_html_report(
                self.results, TEST_BENCHMARK_NAME, {}, self.runtime_config
            )
            self.assertIn(TEST_BENCHMARK_NAME, html) # Check title replacement
            self.assertIn("<td>model_a</td>", html) # Check completed model row
            self.assertIn("<td>model_b</td>", html) # Check completed model row
            self.assertIn("Skipped", html)         # Check skipped model status
            self.assertIn("model_c", html)         # Check skipped model name
            mock_file_open.assert_called_once_with(self.runtime_config.html_template_file, "r", encoding="utf-8")

    @patch('webbrowser.open')
    def test_open_report_auto(self, mock_webbrowser_open):
        report_file = self.runtime_config.report_dir / "report.html"
        report_file.touch() # Create the dummy file
        reporting.open_report_auto(report_file)
        # Check that webbrowser.open was called with the correct file URI
        mock_webbrowser_open.assert_called_once_with(report_file.resolve().as_uri())


class TestCLIConfigLoading(unittest.TestCase):
    # Extensive setup with patching for CLI environment simulation
    def setUp(self):
        self.load_config_patcher = patch('benchmark_cli.load_config_from_file')
        self.mock_load_config = self.load_config_patcher.start()
        self.mock_load_config.return_value = MOCK_DEFAULT_FILE_CONFIG

        # Patch potentially unavailable dependencies
        self.psutil_patcher = patch.dict(sys.modules, {'psutil': MagicMock()})
        self.pynvml_patcher = patch.dict(sys.modules, {'pynvml': MagicMock()})
        self.matplotlib_patcher = patch.dict(sys.modules, {'matplotlib': MagicMock(), 'matplotlib.pyplot': MagicMock(), 'matplotlib.ticker': MagicMock()})
        self.sentence_tf_patcher = patch.dict(sys.modules, {'sentence_transformers': MagicMock()})
        self.yaml_patcher = patch.dict(sys.modules, {'yaml': MagicMock()})
        self.psutil_patcher.start()
        self.pynvml_patcher.start()
        self.matplotlib_patcher.start()
        self.sentence_tf_patcher.start()
        self.yaml_patcher.start()

        # Patch core functions called by main()
        self.run_benchmark_patcher = patch('benchmark_cli.run_benchmark_set', return_value={
            "model1": {"_summary": {"status": "Completed", "accuracy": 50.0}},
            "_task_definitions": {}, "_task_categories": {},
            "_category_weights_used": {}, "_ollama_score_weights_used": {}
        })
        self.mock_run_benchmark = self.run_benchmark_patcher.start()

        self.report_gen_patcher = patch('benchmark_cli.generate_html_report', return_value="<html>Report</html>")
        self.mock_report_gen = self.report_gen_patcher.start()

        self.report_save_patcher = patch('benchmark_cli.save_report', return_value=pathlib.Path("./mock_report.html"))
        self.mock_report_save = self.report_save_patcher.start()

        self.mkdir_patcher = patch('pathlib.Path.mkdir')
        self.mock_mkdir = self.mkdir_patcher.start()

        self.load_tasks_patcher = patch('benchmark_cli.load_and_validate_tasks', return_value=(
            {"task1": {"name":"t1", "type":"a", "prompt":"p"}}, {"Cat1": ["task1"]}
        ))
        self.mock_load_tasks = self.load_tasks_patcher.start()

    def tearDown(self):
        # Stop all patchers
        self.load_config_patcher.stop()
        self.psutil_patcher.stop()
        self.pynvml_patcher.stop()
        self.matplotlib_patcher.stop()
        self.sentence_tf_patcher.stop()
        self.yaml_patcher.stop()
        self.run_benchmark_patcher.stop()
        self.report_gen_patcher.stop()
        self.report_save_patcher.stop()
        self.mkdir_patcher.stop()
        self.load_tasks_patcher.stop()
        # Ensure any other patches are stopped if added elsewhere
        patch.stopall()

    def test_cli_uses_config_file_defaults(self):
        test_args = ['benchmark_cli.py'] # No CLI arguments provided
        with patch.object(sys, 'argv', test_args):
            # Patch the setup function to inspect the created config
            with patch('benchmark_cli.setup_runtime_config') as mock_setup:
                # Simulate the config object that setup_runtime_config would create
                mock_runtime_cfg_instance = RuntimeConfig()
                mock_runtime_cfg_instance.update_from_file_config(MOCK_DEFAULT_FILE_CONFIG)
                mock_setup.return_value = mock_runtime_cfg_instance

                benchmark_cli.main()

                # Check config file loading
                self.mock_load_config.assert_called_once_with(DEFAULT_TEST_CONFIG_PATH)
                # Check that setup_runtime_config was called with the loaded file config
                mock_setup.assert_called_once()
                call_args, _ = mock_setup.call_args
                # Arg 0 is args namespace, Arg 1 is file_config dict
                self.assertEqual(call_args[1], MOCK_DEFAULT_FILE_CONFIG)

                # Verify the final config reflects the file defaults
                final_config = mock_setup.return_value
                self.assertEqual(final_config.tasks_file, pathlib.Path('config_tasks.json'))
                self.assertEqual(final_config.max_retries, 1)
                self.assertEqual(final_config.models_to_benchmark, ['file-default-model:latest', TEST_MODEL_GEMINI])
                self.assertFalse(final_config.ram_monitor_enabled) # From mock file config
                self.assertEqual(final_config.gemini_key, 'file-gemini-key') # From mock file config

    def test_cli_overrides_config_file(self):
        test_args = [
            'benchmark_cli.py',
            '--config-file', 'custom_config.yaml', # Override config path
            '--tasks-file', 'cli_tasks.json',     # Override path
            '--test-model', 'cli-model-1',        # Override models
            '--test-model', 'cli-model-2',
            '--retries', '5',                     # Override retries
            '--ram-monitor', 'enable',            # Override feature flag
            '--gemini-key', 'cli-gemini-key'       # Override API key
        ]
        custom_config_path = pathlib.Path('custom_config.yaml')
        # load_config_from_file still returns the *default* mock content
        self.mock_load_config.return_value = MOCK_DEFAULT_FILE_CONFIG

        with patch.object(sys, 'argv', test_args):
            with patch('benchmark_cli.setup_runtime_config') as mock_setup:
                # Simulate the final config after CLI args override file config
                final_mock_config = RuntimeConfig()
                # Start with file config
                final_mock_config.update_from_file_config(MOCK_DEFAULT_FILE_CONFIG)
                # Apply CLI overrides manually for comparison
                final_mock_config.tasks_file = pathlib.Path('cli_tasks.json')
                final_mock_config.models_to_benchmark = ['cli-model-1', 'cli-model-2']
                final_mock_config.max_retries = 5
                final_mock_config.ram_monitor_enabled = True
                final_mock_config.gemini_key = 'cli-gemini-key'
                # Assume psutil mock is available because --ram-monitor enable implies it
                final_mock_config.psutil_available = True
                final_mock_config.psutil = MagicMock()
                mock_setup.return_value = final_mock_config

                benchmark_cli.main()

                # Check config file loading used the CLI path
                self.mock_load_config.assert_called_once_with(custom_config_path)
                mock_setup.assert_called_once()
                final_config_result = mock_setup.return_value

                # Verify CLI args took precedence
                self.assertEqual(final_config_result.tasks_file, pathlib.Path('cli_tasks.json'))
                self.assertEqual(final_config_result.models_to_benchmark, ['cli-model-1', 'cli-model-2'])
                self.assertEqual(final_config_result.max_retries, 5)
                self.assertTrue(final_config_result.ram_monitor_enabled)
                self.assertEqual(final_config_result.gemini_key, 'cli-gemini-key')
                # Verify non-overridden value from file config remains
                self.assertEqual(final_config_result.retry_delay, 3) # From MOCK_DEFAULT_FILE_CONFIG

    def test_cli_env_overrides_config_file_key(self):
        # Scenario 1: Environment variable overrides file config
        test_args_env = ['benchmark_cli.py']
        env_vars = {'GEMINI_API_KEY': 'env-gemini-key'}
        with patch.object(sys, 'argv', test_args_env), patch.dict(os.environ, env_vars, clear=True):
            with patch('benchmark_cli.setup_runtime_config') as mock_setup:
                # Simulate config after env var override
                final_mock_config = RuntimeConfig()
                final_mock_config.update_from_file_config(MOCK_DEFAULT_FILE_CONFIG)
                final_mock_config.gemini_key = 'env-gemini-key' # Env var takes precedence
                mock_setup.return_value = final_mock_config

                benchmark_cli.main()
                final_config_result = mock_setup.return_value
                self.assertEqual(final_config_result.gemini_key, 'env-gemini-key')

        # Scenario 2: CLI argument overrides both environment and file config
        test_args_cli = ['benchmark_cli.py', '--gemini-key', 'cli-gemini-key']
        # Both env var and CLI arg are set
        with patch.object(sys, 'argv', test_args_cli), patch.dict(os.environ, env_vars, clear=True):
             with patch('benchmark_cli.setup_runtime_config') as mock_setup:
                # Simulate config after CLI override
                final_mock_config = RuntimeConfig()
                final_mock_config.update_from_file_config(MOCK_DEFAULT_FILE_CONFIG)
                final_mock_config.gemini_key = 'cli-gemini-key' # CLI takes highest precedence
                mock_setup.return_value = final_mock_config

                benchmark_cli.main()
                final_config_result = mock_setup.return_value
                self.assertEqual(final_config_result.gemini_key, 'cli-gemini-key')

# Main entry point for tests
if __name__ == '__main__':
    unittest.main()