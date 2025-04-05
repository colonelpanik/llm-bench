# --- System Resource Monitoring (RAM, GPU) ---

import re
# Removed direct requests/json/config imports, use fallback from llm_clients
from llm_clients import _get_ollama_pid_via_api

# --- RAM Monitoring ---

def get_ollama_pids(psutil_module, runtime_config):
    """Identifies PIDs for Ollama processes using psutil, with API fallback."""
    # Args: psutil_module: The imported psutil library. runtime_config (RuntimeConfig): Config for API fallback.
    # Returns: list: A list of potential Ollama process PIDs.
    pids = []
    if psutil_module:
        try:
            NoSuchProcess = getattr(psutil_module, 'NoSuchProcess', Exception)
            AccessDenied = getattr(psutil_module, 'AccessDenied', Exception)

            for proc in psutil_module.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if not proc.is_running(): continue
                    info = proc.info
                    pname = info.get('name', '').lower() if info else ''
                    cmd = ' '.join(info.get('cmdline', [])).lower() if info and info.get('cmdline') else ''

                    # Match common Ollama process names/command lines
                    # Be more specific to avoid matching unrelated processes
                    is_ollama = False
                    if 'ollama' in pname:
                        is_ollama = True
                    elif 'ollama' in cmd and ('serve' in cmd or 'run' in cmd or 'llama_server' in cmd):
                        is_ollama = True

                    if is_ollama:
                        pid = info.get('pid') if info else None
                        if pid and pid not in pids:
                            pids.append(pid)
                except (NoSuchProcess, AccessDenied):
                    continue
                except Exception as inner_e:
                     print(f"[WARN] Error inspecting process {getattr(proc, 'pid', 'N/A')}: {inner_e}")
        except Exception as e:
            print(f"[WARN] Error iterating processes with psutil: {e}")

    # Fallback using ollama show API if psutil fails or finds nothing
    if not pids:
        print("[INFO] No Ollama process found via psutil, trying API fallback...")
        api_pid = _get_ollama_pid_via_api(runtime_config)
        if api_pid:
            if psutil_module and psutil_module.pid_exists(api_pid):
                print(f"[INFO] Found potential Ollama server PID {api_pid} via API and verified with psutil.")
                pids.append(api_pid)
            elif not psutil_module:
                # Cannot verify, but add it anyway if psutil is unavailable
                print(f"[INFO] Found potential PID {api_pid} via API (cannot verify without psutil).")
                pids.append(api_pid)
            else:
                 print(f"[WARN] PID {api_pid} from API error does not seem to exist according to psutil.")
        else:
            print("[INFO] API fallback did not yield a PID.")

    if not pids:
         print("[WARN] Could not determine Ollama process PIDs for RAM monitoring.")
    return pids

def get_combined_rss(pids, psutil_module):
    """Calculates the total RSS memory usage for a list of PIDs."""
    # Args: pids (list): List of process IDs. psutil_module: The imported psutil library.
    # Returns: int: Total RSS memory in bytes, or 0 if unavailable.
    if not psutil_module or not pids:
        return 0
    total_rss = 0
    active_pids = [] # Keep track of PIDs that are still valid
    NoSuchProcess = getattr(psutil_module, 'NoSuchProcess', Exception)
    AccessDenied = getattr(psutil_module, 'AccessDenied', Exception)

    for pid in pids:
        try:
            p = psutil_module.Process(pid)
            # Check if process is still running and accessible
            if p.is_running():
                mem_info = p.memory_info()
                total_rss += mem_info.rss
                active_pids.append(pid)
        except NoSuchProcess:
            pass # Process ended, ignore
        except AccessDenied:
            # Don't warn repeatedly for access denied
            active_pids.append(pid) # Keep PID in case permissions change
        except Exception as e:
            print(f"[WARN] Error getting memory for PID {pid}: {e}")
            active_pids.append(pid) # Keep PID

    # Note: This function doesn't modify the original PID list passed in.
    # Caller might want to update its list based on active_pids if needed.
    return total_rss


# --- GPU Monitoring (NVIDIA Only) ---

def get_gpu_memory_usage(pynvml_module, device_index=0):
    """Gets the used memory for a specific NVIDIA GPU."""
    # Args: pynvml_module: The imported pynvml library. device_index (int): Index of the GPU.
    # Returns: int: Used GPU memory in bytes, or 0 if unavailable/error.
    if not pynvml_module:
        return 0
    try:
        NVMLError = getattr(pynvml_module, 'NVMLError', Exception) # Get NVMLError safely
        handle = pynvml_module.nvmlDeviceGetHandleByIndex(device_index)
        mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used
    except NVMLError as e:
        # Reduce noise: only print specific errors once? Maybe later.
        # print(f"[WARN] Failed to get GPU memory info for device {device_index}: {e}")
        return 0
    except IndexError:
        # print(f"[WARN] GPU device index {device_index} out of range.")
        return 0 # device_index is invalid
    except Exception as e:
        print(f"[WARN] Unexpected error getting GPU memory for device {device_index}: {e}")
        return 0