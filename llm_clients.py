# --- LLM API Client Functions ---

import requests
import json
import sys
import time
import re
import traceback

# Constants like specific API paths (relative to base URL)
OLLAMA_GENERATE_PATH = "/api/generate"
OLLAMA_MODELS_PATH = "/api/tags"
OLLAMA_PULL_PATH = "/api/pull"
OLLAMA_SHOW_PATH = "/api/show"
OLLAMA_PS_PATH = "/api/ps"
GEMINI_API_GENERATE_CONTENT = ":generateContent"

def _get_ollama_url(runtime_config, path):
    """Constructs the full Ollama API URL."""
    return f"{runtime_config.ollama_base_url}{path}"

def _get_gemini_url(runtime_config, model_name):
    """Constructs the full Gemini API URL."""
    model_name_part = model_name.split('/')[-1] if model_name.startswith("models/") else model_name
    base = runtime_config.gemini_api_url_base.rstrip('/')
    return f"{base}/{model_name_part}{GEMINI_API_GENERATE_CONTENT}?key={runtime_config.gemini_key}"


def query_ollama(model_name, prompt, runtime_config):
    """Sends a prompt to a local Ollama model with retry logic."""
    # (Implementation remains the same as previous version)
    url = _get_ollama_url(runtime_config, OLLAMA_GENERATE_PATH)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    if runtime_config.verbose: print(f"    [Verbose] Ollama Payload to {url}: {json.dumps(payload, indent=2)}")

    generated_text = ""
    duration = 0.0
    tokens_per_sec = 0.0
    final_error = None
    attempt_details = []
    request_timeout = runtime_config.request_timeout
    max_retries = runtime_config.max_retries
    retry_delay = runtime_config.retry_delay

    for attempt in range(max_retries + 1):
        start_time = time.time()
        current_error = None
        resp = None
        try:
            resp = requests.post(url, json=payload, timeout=request_timeout)

            if resp.status_code == 404:
                try: e = resp.json().get("error", f"Model {model_name} not found at {url}.")
                except: e = f"404 Status (No JSON body: {resp.text[:100]})"
                current_error = f"404: {e}"
                attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
                break

            if resp.status_code >= 500:
                 current_error = f"Server Error HTTP {resp.status_code} at {url}: {resp.text[:150]}"
                 attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
                 if attempt < max_retries:
                     print(f"    [Retryable Error] {current_error}. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                     time.sleep(retry_delay)
                     continue
                 else: break

            resp.raise_for_status()

            data = resp.json()
            generated_text = data.get("response", "").strip()
            generated_text = re.sub(r'^```[a-zA-Z]*(?:\n)?|(?:\n)?```$', '', generated_text).strip()

            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 0)
            if eval_count > 0 and eval_duration > 0:
                tokens_per_sec = eval_count / (eval_duration / 1e9)
            else:
                tokens_per_sec = None

            duration = time.time() - start_time
            attempt_details.append(f"Attempt {attempt+1}: Success")
            if runtime_config.verbose: print(f"    [Verbose] Ollama Raw Response (Success): {json.dumps(data, indent=2)}")
            final_error = None
            break

        except requests.exceptions.Timeout:
            current_error = f"Timeout after {request_timeout}s connecting to {url}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            if attempt < max_retries:
                 print(f"    [Retryable Error] Request timed out. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                 time.sleep(retry_delay)
                 continue
            else: break
        except requests.exceptions.ConnectionError as e:
            current_error = f"Connection error to {url}: {e}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            if attempt < max_retries:
                 print(f"    [Retryable Error] Connection error. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                 time.sleep(retry_delay)
                 continue
            else: break
        except requests.exceptions.RequestException as e:
            current_error = f"Request failed to {url}: {e}"
            if e.response is not None:
                 current_error += f" Status Code: {e.response.status_code}. Response: {e.response.text[:200]}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            break
        except json.JSONDecodeError as e:
            response_text_snippet = resp.text[:200] if resp else "N/A"
            current_error = f"Failed to decode Ollama response from {url}: {e}. Response text: {response_text_snippet}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            break
        except Exception as e:
            current_error = f"Unexpected error during Ollama query to {url}: {type(e).__name__}: {e}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            if runtime_config.verbose: traceback.print_exc()
            break
        finally:
             if current_error and attempt == max_retries:
                 duration = time.time() - start_time
             final_error = current_error

    if final_error and len(attempt_details) > 1:
         final_error = f"{final_error} (After {len(attempt_details)} attempts: {'; '.join(attempt_details)})"
    elif final_error:
        final_error = f"{final_error} (Attempt 1 failed)"

    return generated_text, duration, tokens_per_sec, final_error


def query_gemini(model_name, prompt, runtime_config):
    """Sends a prompt to Google's Gemini API with retry logic."""
    # (Implementation remains the same as previous version)
    if not runtime_config.gemini_key:
        return "", 0.0, None, "Gemini API Key not provided"

    url = _get_gemini_url(runtime_config, model_name)
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    }
    if runtime_config.verbose: print(f"    [Verbose] Gemini Payload to {url}: {json.dumps(payload, indent=2)}")

    generated_text = ""
    duration = 0.0
    tokens_per_sec = None
    final_error = None
    attempt_details = []
    request_timeout = runtime_config.request_timeout
    max_retries = runtime_config.max_retries
    retry_delay = runtime_config.retry_delay

    for attempt in range(max_retries + 1):
        start_time = time.time()
        current_error = None
        resp = None
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=request_timeout)

            if resp.status_code in [429, 500, 503]:
                 current_error = f"API Error HTTP {resp.status_code} from {url}: {resp.text[:150]}"
                 attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
                 if attempt < max_retries:
                     print(f"    [Retryable Error] {current_error}. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                     time.sleep(retry_delay)
                     continue
                 else: break

            resp.raise_for_status()

            data = resp.json()
            if runtime_config.verbose: print(f"    [Verbose] Gemini Raw Response: {json.dumps(data, indent=2)}")

            prompt_feedback = data.get("promptFeedback", {})
            block_reason_prompt = prompt_feedback.get("blockReason")
            if block_reason_prompt:
                current_error = f"Blocked by API (Prompt): {block_reason_prompt}"
                attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
                break

            candidates = data.get("candidates", [])
            if not candidates:
                 finish_reason_alt = prompt_feedback.get("finishReason")
                 if finish_reason_alt and finish_reason_alt != "FINISH_REASON_UNSPECIFIED":
                      current_error = f"No candidates returned by Gemini; Finish Reason in promptFeedback: {finish_reason_alt}"
                 else:
                      current_error = "No candidates returned by Gemini (and not blocked)"
                 attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
                 break

            candidate = candidates[0]
            content = candidate.get("content", {})
            text_parts = content.get("parts", [])
            if text_parts:
                generated_text = text_parts[0].get("text", "").strip()
                generated_text = re.sub(r'^```[a-zA-Z]*(?:\n)?|(?:\n)?```$', '', generated_text).strip()

            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason not in ["STOP", "MAX_TOKENS"]:
                 print(f"    [API Warning] Gemini finish reason for {model_name}: {finish_reason}")
                 if finish_reason in ["SAFETY", "RECITATION"]:
                     current_error = f"Blocked by API (Response): {finish_reason}"
                     generated_text = ""
                     attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
                     break
                 elif finish_reason == "OTHER":
                      current_error = f"API indicated 'OTHER' finish reason."
                      attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
                      break

            duration = time.time() - start_time
            attempt_details.append(f"Attempt {attempt+1}: Success")
            final_error = None
            break

        except requests.exceptions.Timeout:
            current_error = f"Timeout after {request_timeout}s connecting to {url}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            if attempt < max_retries:
                 print(f"    [Retryable Error] Request timed out. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                 time.sleep(retry_delay)
                 continue
            else: break
        except requests.exceptions.ConnectionError as e:
            current_error = f"Connection error to {url}: {e}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            if attempt < max_retries:
                 print(f"    [Retryable Error] Connection error. Retrying in {retry_delay}s... ({attempt+1}/{max_retries})")
                 time.sleep(retry_delay)
                 continue
            else: break
        except requests.exceptions.RequestException as e:
            try:
                err_detail = e.response.json() if e.response else {}
                api_err_msg = err_detail.get('error', {}).get('message', str(e))
                status_code = e.response.status_code if e.response is not None else 'N/A'
                if status_code == 400 and ("API_KEY_INVALID" in api_err_msg or "API key not valid" in api_err_msg): current_error = "Request failed: Invalid Gemini API Key"
                elif status_code == 404: current_error = f"Request failed: Model '{model_name}' not found or invalid endpoint ({url})"
                elif status_code == 400 and "Unsupported location" in api_err_msg: current_error = f"Request failed: API Key valid but restricted (location?)"
                else: current_error = f"Request failed ({status_code}) to {url}: {api_err_msg}"
            except: current_error = f"Request failed to {url}: {e}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            break
        except json.JSONDecodeError as e:
            response_text_snippet = resp.text[:200] if resp else "N/A"
            current_error = f"Failed to decode Gemini response from {url}: {e}. Response text: {response_text_snippet}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            break
        except Exception as e:
            current_error = f"Unexpected error during Gemini query to {url}: {type(e).__name__}: {e}"
            attempt_details.append(f"Attempt {attempt+1}: Failed ({current_error})")
            if runtime_config.verbose: traceback.print_exc()
            break
        finally:
             if current_error and attempt == max_retries:
                 duration = time.time() - start_time
             final_error = current_error

    if final_error and len(attempt_details) > 1:
         final_error = f"{final_error} (After {len(attempt_details)} attempts: {'; '.join(attempt_details)})"
    elif final_error:
         final_error = f"{final_error} (Attempt 1 failed)"

    if final_error and "Blocked by API" in final_error:
        generated_text = ""

    return generated_text, duration, tokens_per_sec, final_error


def pull_ollama_model(model_name, runtime_config):
    """Attempts to pull an Ollama model using the API."""
    # (Implementation remains the same as previous version)
    url = _get_ollama_url(runtime_config, OLLAMA_PULL_PATH)
    pull_timeout = 3600

    print(f"Pulling model '{model_name}' from remote registry (via {url})... Timeout: {pull_timeout}s. This may take a while.")
    try:
        with requests.post(url, json={"name": model_name, "stream": True}, stream=True, timeout=pull_timeout) as resp:
            if resp.status_code != 200:
                try: detail = resp.json().get("error", f"HTTP {resp.status_code}")
                except: detail = resp.text[:200]
                print(f"[ERROR] Pull failed for '{model_name}': {detail}")
                return False

            last_status_line = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        status = chunk.get("status", "")
                        prog_detail = status
                        if 'completed' in chunk and 'total' in chunk and chunk['total'] > 0:
                             prog_detail += f" ({chunk['completed'] / (1024*1024):.1f}/{chunk['total'] / (1024*1024):.1f} MB)"
                        elif 'digest' in chunk:
                             prog_detail += f" (Digest: {chunk['digest'][:12]}...)"
                        if prog_detail != last_status_line:
                             print(f"  Progress: {prog_detail}", end='\r')
                             last_status_line = prog_detail
                    except json.JSONDecodeError: pass
                    except Exception as stream_err: print(f"\n[WARN] Error processing pull stream line: {stream_err}")

            print(f"\n[SUCCESS] Pull stream completed for model '{model_name}'.")
            return True

    except requests.exceptions.Timeout:
        print(f"\n[ERROR] Pull request for '{model_name}' timed out after {pull_timeout}s.")
        return False
    except requests.exceptions.ConnectionError as e:
         print(f"\n[ERROR] Connection error during pull for '{model_name}' to {url}: {e}")
         return False
    except Exception as e:
        print(f"\n[ERROR] Pull request failed for '{model_name}': {e}")
        return False


def get_local_ollama_models(runtime_config):
    """Fetches the list of locally available Ollama models via API."""
    # (Implementation remains the same as previous version)
    url = _get_ollama_url(runtime_config, OLLAMA_MODELS_PATH)
    local_models = set()
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        models_data = r.json().get("models", [])
        for m in models_data:
            if m_name := m.get("name"):
                local_models.add(m_name)
        return local_models
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.RequestException as e:
        print(f"[WARN] get_local_ollama_models: API request failed to {url}: {e}")
        return set()
    except Exception as e:
        print(f"[WARN] get_local_ollama_models: Failed to parse models list from {url}: {e}")
        return set()

def get_ollama_running_models(runtime_config):
    """Gets the list of models currently loaded in Ollama memory via /api/ps."""
    # (Implementation remains the same as previous version)
    url = _get_ollama_url(runtime_config, OLLAMA_PS_PATH)
    running_models = []
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 404:
            print(f"[INFO] Ollama API endpoint {OLLAMA_PS_PATH} not found. Cannot check running models.")
            return []
        r.raise_for_status()
        data = r.json()
        models_info = data.get("models", [])
        for model_info in models_info:
            if name := model_info.get("name"):
                running_models.append(name)
        return running_models
    except requests.exceptions.ConnectionError:
        print(f"[WARN] get_ollama_running_models: Could not connect to Ollama API at {url}.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[WARN] get_ollama_running_models: API request failed to {url}: {e}")
        return None
    except Exception as e:
        print(f"[WARN] get_ollama_running_models: Failed to parse running models list from {url}: {e}")
        return None

def unload_ollama_model(model_name, runtime_config):
    """Sends a request to Ollama to unload a specific model (using keep_alive=0)."""
    # Args: model_name (str): The model tag to unload.
    #       runtime_config (RuntimeConfig): Config object for URL and timeout.
    # Returns: bool: True if the request was sent successfully (status 200), False otherwise.
    url = _get_ollama_url(runtime_config, OLLAMA_GENERATE_PATH)
    payload = {
        "model": model_name,
        "prompt": "",       # Empty prompt
        "keep_alive": 0,    # Request unload
        "stream": False
    }
    # Use a shorter timeout for unload requests? Default request timeout might be too long.
    unload_timeout = 15 # seconds

    if runtime_config.verbose: print(f"    [Verbose] Sending unload request for {model_name} to {url}")

    try:
        resp = requests.post(url, json=payload, timeout=unload_timeout)
        if resp.status_code == 200:
            if runtime_config.verbose: print(f"    [Verbose] Unload request for {model_name} successful (status 200).")
            return True
        elif resp.status_code == 404:
             # Model might already be unloaded or never existed
             if runtime_config.verbose: print(f"    [Verbose] Model {model_name} not found during unload attempt (status 404). Assuming unloaded.")
             return True # Treat as success in terms of desired state
        else:
            print(f"[WARN] Unload request for model '{model_name}' failed with status {resp.status_code}: {resp.text[:150]}")
            return False
    except requests.exceptions.Timeout:
        print(f"[WARN] Unload request for model '{model_name}' timed out after {unload_timeout}s.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Unload request failed for model '{model_name}': {e}")
        return False
    except Exception as e:
        print(f"[WARN] Unexpected error during unload request for '{model_name}': {e}")
        return False


# --- Fallback for PID detection (less reliable than psutil) ---
def _get_ollama_pid_via_api(runtime_config):
    """Tries to get Ollama PID via a failed API call (fallback)."""
    # (Implementation remains the same as previous version)
    url = _get_ollama_url(runtime_config, OLLAMA_SHOW_PATH)
    try:
        dummy_model = "nonexistent_model_for_pid_check:latest"
        show_resp = requests.post(url, json={"name": dummy_model, "verbose": True}, timeout=5)
        if show_resp.status_code == 404:
            try:
                show_data = show_resp.json()
                error_msg = show_data.get("error", "")
                pid_match = re.search(r'process with pid (\d+)', error_msg)
                if pid_match:
                    return int(pid_match.group(1))
            except: pass
    except: pass
    return None