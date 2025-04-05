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
import os
from unittest.mock import patch, MagicMock, mock_open, ANY
import requests
import yaml
import re

# Add project root to path to allow importing modules
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Modules to test (import after adding to path)
import config
from config import RuntimeConfig, DEFAULT_OLLAMA_BASE_URL, DEFAULT_GEMINI_API_URL_BASE # Import class and defaults
import utils
from utils import TimeoutException, HAS_SIGNAL_ALARM, truncate_text
import llm_clients # Import module itself
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
DEFAULT_TEST_CONFIG_PATH = pathlib.Path("./config.yaml") # Used in CLI tests

# --- Mock Responses ---
# (Mock responses remain the same as before)
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
MOCK_OLLAMA_PS_RESPONSE_EMPTY = {"models": []}
MOCK_OLLAMA_PS_RESPONSE_RUNNING = {"models": [{"name": "preloaded-model:latest", "size": 123, "expires_at": "..."}]}
MOCK_OLLAMA_UNLOAD_RESPONSE = {"status": "success"}

# --- Mock Default Config Data (Updated Structure) ---
# (Mock config data remains the same)
MOCK_DEFAULT_FILE_CONFIG = {
    'api': {'gemini_api_key': 'file-gemini-key', 'ollama_host_url': 'http://file-config-host:11434'},
    'default_models': ['file-default-model:latest', TEST_MODEL_GEMINI],
    'code_models': ['file-code-model:latest'],
    'paths': { 'tasks_file': 'config_tasks.json', 'report_dir': './config_report', 'cache_dir': './config_cache', 'template_file': 'config_template.html' },
    'timeouts': {'request': 200, 'code_execution_per_case': 15},
    'retries': {'max_retries': 1, 'retry_delay': 3},
    'cache': {'ttl_seconds': 3600},
    'scoring': { 'ollama_perf_score': {'accuracy': 0.6, 'tokens_per_sec': 0.2, 'ram_efficiency': 0.2}, 'category_weights': {'General NLP': 0.9, 'Code Generation': 1.1, 'default': 0.6}},
    'features': { 'ram_monitor': False, 'gpu_monitor': False, 'visualizations': False, 'semantic_eval': False },
    'evaluation': {'default_min_confidence': 0.8, 'passing_score_threshold': 75.0}
}


# --- Mock Objects ---
# (MockPsutilProcess remains the same)
class MockPsutilProcess:
    def __init__(self, pid, name='python', cmdline=None, rss=100 * 1024 * 1024):
        self.pid = pid
        self._name = name
        self._cmdline = cmdline or ['python']
        self._rss = rss
        self._is_running = True
        self.info = {'pid': self.pid, 'name': self._name, 'cmdline': self._cmdline}

    def memory_info(self):
        # Assume psutil is mocked if this is called in a test
        psutil_mock = MagicMock()
        psutil_mock.NoSuchProcess = Exception
        if not self._is_running:
            raise psutil_mock.NoSuchProcess(self.pid)
        mock_mem = MagicMock()
        mock_mem.rss = self._rss
        return mock_mem

    def is_running(self):
        return self._is_running

    def terminate(self):
        self._is_running = False

# --- Base Mock RuntimeConfig for Tests ---
# ** FIX: Initialize mock libraries correctly **
class BaseMockRuntimeConfig(RuntimeConfig):
     def __init__(self, psutil_available=False, pynvml_available=False, yaml_available=True, matplotlib_available=False, sentence_transformers_available=False, **kwargs):
         super().__init__()
         # Apply minimal valid defaults needed for tests
         self.ollama_base_url = DEFAULT_OLLAMA_BASE_URL
         self.gemini_api_url_base = DEFAULT_GEMINI_API_URL_BASE
         self.gemini_key = "mock-test-key"
         self.request_timeout = 30
         self.max_retries = 1
         self.retry_delay = 1
         self.code_exec_timeout = 5
         self.passing_score_threshold = 70.0
         self.default_min_confidence = 0.75

         # Set availability flags based on constructor args
         self.pyyaml_available = yaml_available
         self.psutil_available = psutil_available
         self.pynvml_available = pynvml_available
         self.matplotlib_available = matplotlib_available
         self.sentence_transformers_available = sentence_transformers_available

         # Assign mock objects *if* the library is marked as available
         self.yaml = MagicMock() if yaml_available else None
         self.psutil = MagicMock() if psutil_available else None
         self.pynvml = MagicMock() if pynvml_available else None
         self.matplotlib = MagicMock() if matplotlib_available else None
         self.plt = MagicMock() if matplotlib_available else None
         self.mticker = MagicMock() if matplotlib_available else None
         self.SentenceTransformer = MagicMock() if sentence_transformers_available else None
         self.st_util = MagicMock() if sentence_transformers_available else None
         self.semantic_model = MagicMock() if sentence_transformers_available else None

         # Configure mocks if they were created
         if yaml_available: self.yaml.YAMLError = yaml.YAMLError if 'yaml' in sys.modules else Exception
         if psutil_available: self.psutil.NoSuchProcess = Exception; self.psutil.AccessDenied = Exception; self.psutil.pid_exists.return_value=True; self.psutil.process_iter.return_value=[]
         if pynvml_available: self.pynvml.NVMLError = Exception; self.pynvml.nvmlInit.return_value=None; self.pynvml.nvmlShutdown.return_value=None; self.pynvml.nvmlDeviceGetCount.return_value = 1 # Assume 1 GPU if mocked
         if matplotlib_available: self.plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock())) # fig, ax
         if sentence_transformers_available:
              mock_tensor = MagicMock(); mock_tensor.item.return_value = 0.8
              self.semantic_model.encode.return_value = "mock_embedding"
              self.st_util.pytorch_cos_sim.return_value = mock_tensor


         # Apply any other overrides passed via kwargs
         for key, value in kwargs.items():
             if hasattr(self, key):
                 setattr(self, key, value)

# --- Test Classes ---

class TestUtils(unittest.TestCase):
    # (Tests remain the same)
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
        # Use the base mock config
        self.mock_runtime_config = BaseMockRuntimeConfig(gemini_key="test-key")

    # (All test cases remain the same, they already use self.mock_runtime_config)
    @patch('requests.post')
    def test_query_ollama_success(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_OLLAMA_RESPONSE_SUCCESS
        mock_post.return_value = mock_resp
        expected_url = f"{self.mock_runtime_config.ollama_base_url}{llm_clients.OLLAMA_GENERATE_PATH}"
        text, duration, tokps, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA, "test prompt", self.mock_runtime_config)
        mock_post.assert_called_once_with(expected_url, json=ANY, timeout=self.mock_runtime_config.request_timeout)
        self.assertEqual(text, "This is a successful test response.")
        self.assertIsNone(error)
        self.assertGreater(duration, 0)
        self.assertAlmostEqual(tokps, 10.0, delta=0.1)

    @patch('requests.post')
    def test_query_ollama_404_error(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 404
        mock_resp.json.return_value = {"error": "model not found"}
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_ollama("missing-model", "p", self.mock_runtime_config)
        self.assertEqual(text, "")
        self.assertIsNotNone(error); self.assertIn("404", error); self.assertIn("model not found", error)

    @patch('requests.post')
    @patch('time.sleep', return_value=None)
    def test_query_ollama_500_retry_fail(self, mock_sleep, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 500; mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA, "p", self.mock_runtime_config)
        self.assertEqual(text, "")
        self.assertIsNotNone(error); self.assertIn("Server Error HTTP 500", error)
        self.assertIn(f"After {self.mock_runtime_config.max_retries + 1} attempts", error)
        self.assertEqual(mock_post.call_count, self.mock_runtime_config.max_retries + 1)
        self.assertEqual(mock_sleep.call_count, self.mock_runtime_config.max_retries)

    @patch('requests.post')
    @patch('time.sleep', return_value=None)
    def test_query_ollama_timeout_retry_success(self, mock_sleep, mock_post):
        mock_resp_success = MagicMock(); mock_resp_success.status_code = 200
        mock_resp_success.json.return_value = MOCK_OLLAMA_RESPONSE_SUCCESS
        mock_post.side_effect = [requests.exceptions.Timeout("Request timed out"), mock_resp_success]
        text, duration, tokps, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA, "p", self.mock_runtime_config)
        self.assertEqual(text, "This is a successful test response.")
        self.assertIsNone(error)
        self.assertEqual(mock_post.call_count, 2); self.assertEqual(mock_sleep.call_count, 1)

    @patch('requests.post')
    def test_query_gemini_success(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_SUCCESS
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, "This is a successful Gemini test response.")
        self.assertIsNone(error); self.assertIsNone(tokps); self.assertGreater(duration, 0)

    @patch('requests.post')
    def test_query_gemini_blocked_prompt(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_BLOCKED
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, ""); self.assertIsNotNone(error); self.assertIn("Blocked by API (Prompt): SAFETY", error)

    @patch('requests.post')
    def test_query_gemini_blocked_candidate(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_CANDIDATE_BLOCKED
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, ""); self.assertIsNotNone(error); self.assertIn("Blocked by API (Response): SAFETY", error)

    @patch('requests.get', side_effect=requests.exceptions.ConnectionError("Connection failed"))
    def test_get_local_ollama_models_connection_error(self, mock_get):
        models = llm_clients.get_local_ollama_models(self.mock_runtime_config)
        self.assertIsNone(models)

    @patch('requests.get')
    def test_get_local_ollama_models(self, mock_get):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_OLLAMA_MODELS_RESPONSE
        mock_get.return_value = mock_resp
        expected_url = f"{self.mock_runtime_config.ollama_base_url}{llm_clients.OLLAMA_MODELS_PATH}"
        models = llm_clients.get_local_ollama_models(self.mock_runtime_config)
        self.assertEqual(models, {TEST_MODEL_OLLAMA, "another-model:7b"})
        mock_get.assert_called_once_with(expected_url, timeout=10)

    @patch('requests.get')
    def test_get_ollama_running_models_empty(self, mock_get):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_OLLAMA_PS_RESPONSE_EMPTY
        mock_get.return_value = mock_resp
        expected_url = f"{self.mock_runtime_config.ollama_base_url}{llm_clients.OLLAMA_PS_PATH}"
        models = llm_clients.get_ollama_running_models(self.mock_runtime_config)
        self.assertEqual(models, [])
        mock_get.assert_called_once_with(expected_url, timeout=5)

    @patch('requests.get')
    def test_get_ollama_running_models_found(self, mock_get):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_OLLAMA_PS_RESPONSE_RUNNING
        mock_get.return_value = mock_resp
        expected_url = f"{self.mock_runtime_config.ollama_base_url}{llm_clients.OLLAMA_PS_PATH}"
        models = llm_clients.get_ollama_running_models(self.mock_runtime_config)
        self.assertEqual(models, ["preloaded-model:latest"])
        mock_get.assert_called_once_with(expected_url, timeout=5)

    @patch('requests.post')
    def test_unload_ollama_model_success(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        expected_url = f"{self.mock_runtime_config.ollama_base_url}{llm_clients.OLLAMA_GENERATE_PATH}"
        success = llm_clients.unload_ollama_model("model-to-unload", self.mock_runtime_config)
        self.assertTrue(success)
        mock_post.assert_called_once_with(expected_url, json=ANY, timeout=15) # Check default unload timeout

    @patch('requests.post')
    def test_unload_ollama_model_404(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 404
        mock_post.return_value = mock_resp
        success = llm_clients.unload_ollama_model("model-not-loaded", self.mock_runtime_config)
        self.assertTrue(success)

    @patch('requests.post', side_effect=requests.exceptions.Timeout)
    def test_unload_ollama_model_timeout(self, mock_post):
        success = llm_clients.unload_ollama_model("model-to-unload", self.mock_runtime_config)
        self.assertFalse(success)


class TestSystemMonitor(unittest.TestCase):
    def setUp(self):
        # Setup mock config, enable psutil for these tests
        self.mock_runtime_config_psutil = BaseMockRuntimeConfig(psutil_available=True)
        self.mock_runtime_config_no_psutil = BaseMockRuntimeConfig(psutil_available=False)
        self.mock_runtime_config_gpu = BaseMockRuntimeConfig(pynvml_available=True)

    def test_get_ollama_pids_found(self):
        mock_proc1 = MockPsutilProcess(pid=123, name='ollama', cmdline=['ollama', 'serve'])
        mock_proc2 = MockPsutilProcess(pid=456, name='other')
        # Use the psutil mock from the config object
        mock_psutil_mod = self.mock_runtime_config_psutil.psutil
        mock_psutil_mod.process_iter.return_value = [mock_proc1, mock_proc2]

        # Pass the config object
        pids = system_monitor.get_ollama_pids(mock_psutil_mod, self.mock_runtime_config_psutil)
        self.assertEqual(pids, [123])
        mock_psutil_mod.process_iter.assert_called_once_with(['pid', 'name', 'cmdline'])

    # Patch the specific fallback function *within llm_clients* as that's where it lives
    @patch('llm_clients._get_ollama_pid_via_api', return_value=789)
    @patch('llm_clients.get_ollama_running_models', return_value=[])
    def test_get_ollama_pids_fallback(self, mock_get_running, mock_api_fallback):
        # Use config where psutil is available but finds nothing
        mock_psutil_mod = self.mock_runtime_config_psutil.psutil
        mock_psutil_mod.process_iter.return_value = [] # Find nothing
        mock_psutil_mod.pid_exists.return_value = True

        # Pass the config object
        pids = system_monitor.get_ollama_pids(mock_psutil_mod, self.mock_runtime_config_psutil)
        self.assertEqual(pids, [789])
        mock_api_fallback.assert_called_once_with(self.mock_runtime_config_psutil)
        mock_get_running.assert_not_called()

    @patch('llm_clients._get_ollama_pid_via_api', return_value=789)
    @patch('llm_clients.get_ollama_running_models', return_value=[])
    def test_get_ollama_pids_no_psutil_no_fallback(self, mock_get_running, mock_api_fallback):
        # Use config where psutil is unavailable
        pids = system_monitor.get_ollama_pids(None, self.mock_runtime_config_no_psutil)
        self.assertEqual(pids, [789]) # Should return empty list
        mock_api_fallback.assert_called_once_with(self.mock_runtime_config_no_psutil)
        mock_get_running.assert_not_called()


    def test_get_combined_rss(self):
        mock_proc1 = MockPsutilProcess(pid=1, rss=100 * 1024**2)
        mock_proc2 = MockPsutilProcess(pid=2, rss=200 * 1024**2)
        mock_proc3 = MockPsutilProcess(pid=3, rss=50 * 1024**2); mock_proc3.terminate()
        # Use the psutil mock from the config object
        mock_psutil_mod = self.mock_runtime_config_psutil.psutil
        mock_psutil_mod.Process.side_effect = lambda pid: {1: mock_proc1, 2: mock_proc2, 3: mock_proc3}[pid]

        total_rss = system_monitor.get_combined_rss([1, 2, 3], mock_psutil_mod)
        self.assertEqual(total_rss, 300 * 1024**2)
        self.assertEqual(mock_psutil_mod.Process.call_count, 3)

    def test_get_gpu_memory_usage_success(self):
        mock_handle = MagicMock()
        mock_mem_info = MagicMock()
        mock_mem_info.used = 5 * 1024**3
        mock_nvml_mod = self.mock_runtime_config_gpu.pynvml # Get mock from config
        mock_nvml_mod.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml_mod.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        mem_used = system_monitor.get_gpu_memory_usage(mock_nvml_mod, device_index=0)
        self.assertEqual(mem_used, 5 * 1024**3)
        mock_nvml_mod.nvmlDeviceGetHandleByIndex.assert_called_with(0)
        mock_nvml_mod.nvmlDeviceGetMemoryInfo.assert_called_with(mock_handle)

    def test_get_gpu_memory_usage_error(self):
        mock_nvml_mod = self.mock_runtime_config_gpu.pynvml
        if not hasattr(mock_nvml_mod, 'NVMLError'): mock_nvml_mod.NVMLError = Exception
        mock_nvml_mod.nvmlDeviceGetHandleByIndex.side_effect = mock_nvml_mod.NVMLError("Device not found")
        mem_used = system_monitor.get_gpu_memory_usage(mock_nvml_mod, device_index=0)
        self.assertEqual(mem_used, 0)


class TestEvaluation(unittest.TestCase):
    # Use BaseMockRuntimeConfig for simplified setup
    def setUp(self):
        self.runtime_config = BaseMockRuntimeConfig(
            psutil_available=True, pynvml_available=True, yaml_available=True,
            matplotlib_available=True, sentence_transformers_available=True
        )
        self.runtime_config_no_yaml = BaseMockRuntimeConfig(yaml_available=False)
        self.runtime_config_no_semantic = BaseMockRuntimeConfig(sentence_transformers_available=False)

    # (Evaluation tests remain the same)
    def test_evaluate_response_dispatch_code(self):
        task = {"name": "t", "type": "code_generation", "prompt": "p", "function_name": "f", "test_cases": []}
        with patch('evaluation.evaluator.execute_and_test_code') as mock_exec:
            mock_exec.return_value = (True, "Code OK")
            metric, details = evaluation.evaluate_response(task, "code", self.runtime_config)
            mock_exec.assert_called_once_with("code", "f", [], self.runtime_config.code_exec_timeout)
            self.assertTrue(metric); self.assertEqual(details, "Code OK")

    def test_evaluate_response_dispatch_semantic(self):
        task = {"name": "t", "type": "summarization", "evaluation_method": "semantic", "prompt": "p", "expected_keywords": ["ref"]}
        with patch('evaluation.evaluator.evaluate_semantic_similarity') as mock_sem:
            mock_sem.return_value = (95.0, "Similarity High")
            metric, details = evaluation.evaluate_response(task, "response", self.runtime_config)
            mock_sem.assert_called_once_with(task, "response", self.runtime_config.semantic_model, self.runtime_config.st_util, ANY)
            self.assertEqual(metric, 95.0); self.assertEqual(details, "Similarity High")

    def test_evaluate_response_dispatch_semantic_unavailable(self):
        task = {"name": "t", "type": "summarization", "evaluation_method": "semantic", "prompt": "p", "expected_keywords": ["ref"]}
        with patch('evaluation.evaluator.partial_keyword_scoring') as mock_partial:
            mock_partial.return_value = (50.0, "Partial Match")
            metric, details = evaluation.evaluate_response(task, "response", self.runtime_config_no_semantic)
            mock_partial.assert_called_once_with(task, "response")
            self.assertEqual(metric, 50.0); self.assertEqual(details, "Partial Match")

    # Other evaluation tests...


class TestScoring(unittest.TestCase):
    # (Scoring test remains the same)
    def test_compute_scores(self):
        mock_results = {
            "ollama_fast_good": {"_summary": {"model_name": "ollama_fast_good", "provider": "ollama", "status": "Completed", "accuracy": 90.0, "tokens_per_sec_avg": 100.0, "delta_ram_mb": 500.0, "per_type": {"General NLP": {"accuracy": 90.0, "count": 10, "success_api_calls": 10}}}},
            "ollama_slow_bad": {"_summary": {"model_name": "ollama_slow_bad", "provider": "ollama", "status": "Completed", "accuracy": 60.0, "tokens_per_sec_avg": 20.0, "delta_ram_mb": 1500.0, "per_type": {"General NLP": {"accuracy": 60.0, "count": 10, "success_api_calls": 10}}}},
            "gemini_ok": {"_summary": {"model_name": "gemini_ok", "provider": "gemini", "status": "Completed", "accuracy": 75.0, "per_type": {"General NLP": {"accuracy": 75.0, "count": 10, "success_api_calls": 10}}}},
            "skipped_model": {"_summary": {"model_name": "skipped_model", "status": "Skipped"}},
            "_task_categories": {"General NLP": ["task1"]}, "_task_definitions": {"task1": {"type": "General NLP"}},
            "_ollama_score_weights_used": {"accuracy": 0.5, "tokens_per_sec": 0.3, "ram_efficiency": 0.2}
        }
        category_weights = {"General NLP": 1.0, "default": 0.5}; default_weight = 0.5
        scoring.compute_performance_scores(mock_results, category_weights, default_weight)
        score1 = mock_results["ollama_fast_good"]["_summary"]["ollama_perf_score"]
        score2 = mock_results["ollama_slow_bad"]["_summary"]["ollama_perf_score"]
        self.assertIsNotNone(score1); self.assertIsNotNone(score2); self.assertGreater(score1, score2)
        self.assertAlmostEqual(mock_results["ollama_fast_good"]["_summary"]["overall_weighted_score"], 90.0)
        self.assertAlmostEqual(mock_results["ollama_slow_bad"]["_summary"]["overall_weighted_score"], 60.0)
        self.assertAlmostEqual(mock_results["gemini_ok"]["_summary"]["overall_weighted_score"], 75.0)
        self.assertIsNone(mock_results["skipped_model"]["_summary"].get("ollama_perf_score"))
        self.assertIsNone(mock_results["skipped_model"]["_summary"].get("overall_weighted_score"))
        #check that the ollama_perf_score is not none
        self.assertIsNotNone(mock_results["ollama_fast_good"]["_summary"]["ollama_perf_score"])
        self.assertIsNotNone(mock_results["ollama_slow_bad"]["_summary"]["ollama_perf_score"])
        #check that the ollama_perf_score is not 0
        self.assertNotEqual(mock_results["ollama_fast_good"]["_summary"]["ollama_perf_score"], 0.0)
        self.assertNotEqual(mock_results["ollama_slow_bad"]["_summary"]["ollama_perf_score"], 0.0)


class TestCacheManager(unittest.TestCase):
    # (Tests remain the same)
    def setUp(self): 
        self.temp_dir=tempfile.mkdtemp(); self.cache_dir=pathlib.Path(self.temp_dir); safe_name=re.sub(r"[^a-zA-Z0-9_\-]+", "_", TEST_BENCHMARK_NAME.lower()); self.cache_file=self.cache_dir / f"cache_{safe_name}.json"; self.cache_ttl=3600
    def tearDown(self): 
        shutil.rmtree(self.temp_dir)
    def test_get_cache_path(self): 
        path=cache_manager.get_cache_path(self.cache_dir, TEST_BENCHMARK_NAME); self.assertEqual(path, self.cache_file)
    def test_save_load_cache(self): 
        data={"m1": {"_summary": {"acc": 80.0}}}; cache_manager.save_cache(self.cache_file, data); self.assertTrue(self.cache_file.is_file()); loaded=cache_manager.load_cache(self.cache_file, self.cache_ttl); self.assertEqual(loaded, data)
    def test_load_cache_nonexistent(self): 
        loaded=cache_manager.load_cache(self.cache_file, self.cache_ttl); self.assertIsNone(loaded)
    def test_load_cache_expired(self): 
        data={"m1": {"_summary": {"acc": 80.0}}}; cache_manager.save_cache(self.cache_file, data); 
        with patch('time.time', return_value=time.time()+self.cache_ttl+10): 
            loaded=cache_manager.load_cache(self.cache_file, self.cache_ttl); self.assertIsNone(loaded)
    def test_load_cache_corrupted(self): 
        with open(self.cache_file, "w") as f: 
            f.write("{invalid json")
            loaded=cache_manager.load_cache(self.cache_file, self.cache_ttl)
            self.assertIsNone(loaded)
    def test_clear_cache(self): 
        data={"m1": {"_summary": {"acc": 80.0}}}
        cache_manager.save_cache(self.cache_file, data)
        self.assertTrue(self.cache_file.is_file()); cache_manager.clear_cache(self.cache_file)
        self.assertFalse(self.cache_file.is_file())

class TestReporting(unittest.TestCase):
    # (Tests remain the same)
    def setUp(self): 
        self.temp_dir_obj=tempfile.TemporaryDirectory()
        self.temp_dir=self.temp_dir_obj.name
        self.runtime_config=BaseMockRuntimeConfig(matplotlib_available=True, visualizations_enabled=True)
        self.runtime_config.report_dir=pathlib.Path(self.temp_dir)
        self.runtime_config.report_img_dir=self.runtime_config.report_dir/"images"; 
        self.runtime_config.report_img_dir.mkdir(parents=True, exist_ok=True); 
        self.runtime_config.html_template_file=pathlib.Path("./report_template.html"); 
        self.runtime_config_no_plots=BaseMockRuntimeConfig(matplotlib_available=False)
        self.results={"m_a":{"_summary":{"status":"Completed","model_name":"m_a","provider":"ollama","accuracy":90.0,"overall_weighted_score":90.0}}, "m_b":{"_summary":{"status":"Completed","model_name":"m_b","provider":"ollama","accuracy":80.0,"overall_weighted_score":80.0,"ollama_perf_score":75.0}}, "m_c":{"_summary":{"status":"Skipped","model_name":"m_c"}},"_t_def":{},"_t_cat":{},"_cat_w":{}}
        if self.runtime_config.matplotlib_available: 
            self.runtime_config.plt=MagicMock()
            self.runtime_config.plt.savefig=MagicMock()
            self.runtime_config.plt.subplots=MagicMock(return_value=(MagicMock(),MagicMock()))
        self.runtime_config.plt.close=MagicMock()
    def tearDown(self): self.temp_dir_obj.cleanup()
    def test_generate_ranking_plot_success(self):
        if not self.runtime_config.matplotlib_available: self.skipTest("Matplotlib mock unavailable")
        path=reporting.generate_ranking_plot(self.results,"accuracy","Acc Plot","Acc","acc.png",self.runtime_config)
        self.assertEqual(path,"images/acc.png"); self.runtime_config.plt.subplots.assert_called(); save_path=self.runtime_config.report_img_dir/"acc.png"; self.runtime_config.plt.savefig.assert_called_once_with(save_path, bbox_inches='tight', dpi=100); self.runtime_config.plt.close.assert_called_once()

class TestCLIConfigLoading(unittest.TestCase):
    # (Tests remain the same)
    def setUp(self): self.load_config_patcher=patch('benchmark_cli.load_config_from_file', return_value=MOCK_DEFAULT_FILE_CONFIG); self.mock_load_config=self.load_config_patcher.start(); self.setup_config_patcher=patch('benchmark_cli.setup_runtime_config'); self.mock_setup_config=self.setup_config_patcher.start(); self.mock_setup_config.return_value=BaseMockRuntimeConfig(); self.run_benchmark_patcher=patch('benchmark_cli.run_benchmark_set', return_value={"m1":{"_summary":{"status":"Completed"}}}); self.mock_run_benchmark=self.run_benchmark_patcher.start(); self.report_gen_patcher=patch('benchmark_cli.generate_html_report', return_value="<html></html>"); self.mock_report_gen=self.report_gen_patcher.start(); self.report_save_patcher=patch('benchmark_cli.save_report', return_value=pathlib.Path(".")); self.mock_report_save=self.report_save_patcher.start(); self.mkdir_patcher=patch('pathlib.Path.mkdir'); self.mock_mkdir=self.mkdir_patcher.start(); self.load_tasks_patcher=patch('benchmark_cli.load_and_validate_tasks', return_value=({"t1":{}},{"C1":["t1"]})); self.mock_load_tasks=self.load_tasks_patcher.start(); self.check_dep_patcher=patch('benchmark_cli.check_dependencies'); self.mock_check_dep=self.check_dep_patcher.start(); self.get_local_models_patcher=patch('llm_clients.get_local_ollama_models', return_value={'file-default-model:latest'}); self.mock_get_local_models=self.get_local_models_patcher.start(); self.get_running_models_patcher=patch('llm_clients.get_ollama_running_models', return_value=[]); self.mock_get_running_models=self.get_running_models_patcher.start(); self.unload_model_patcher=patch('llm_clients.unload_ollama_model', return_value=True); self.mock_unload_model=self.unload_model_patcher.start()
    def tearDown(self): patch.stopall()
    def test_cli_uses_config_file_defaults(self):
        test_args=['benchmark_cli.py']; self.setup_config_patcher.stop()
        with patch.object(sys,'argv',test_args): benchmark_cli.main()
        self.mock_load_config.assert_called_once_with(DEFAULT_TEST_CONFIG_PATH); self.mock_run_benchmark.assert_called_once()
        _, call_kwargs=self.mock_run_benchmark.call_args; final_config=call_kwargs.get('runtime_config')
        self.assertEqual(final_config.gemini_key,'file-gemini-key'); self.assertEqual(final_config.tasks_file,pathlib.Path('config_tasks.json')); self.assertEqual(final_config.max_retries,1); self.assertEqual(final_config.ollama_base_url,'http://file-config-host:11434'); self.assertEqual(final_config.models_to_benchmark,['file-default-model:latest',TEST_MODEL_GEMINI]); self.assertEqual(final_config.code_models_to_benchmark,['file-code-model:latest']); self.assertFalse(final_config.ram_monitor_enabled)
        self.setup_config_patcher.start()
    @patch.dict(os.environ,{"GEMINI_API_KEY":"env-key","OLLAMA_HOST":"http://env-host:11434"},clear=True)
    def test_cli_overrides_env_overrides_file(self):
        test_args=['benchmark_cli.py','--gemini-key','cli-key','--test-model','cli-model']; self.setup_config_patcher.stop()
        with patch.object(sys,'argv',test_args): benchmark_cli.main()
        self.mock_run_benchmark.assert_called_once(); _, call_kwargs=self.mock_run_benchmark.call_args; final_config=call_kwargs.get('runtime_config')
        self.assertEqual(final_config.gemini_key,'cli-key'); self.assertEqual(final_config.ollama_base_url,'http://env-host:11434'); self.assertEqual(final_config.models_to_benchmark,['cli-model']); self.assertEqual(final_config.code_models_to_benchmark,['cli-model'])
        self.setup_config_patcher.start()

# Main entry point for tests
if __name__ == '__main__':
    unittest.main()