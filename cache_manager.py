# --- Cache Management Logic ---

import json
import time
import re
from datetime import datetime
# Removed direct import of CACHE_TTL_SECONDS from config

def get_cache_path(cache_dir, benchmark_name):
    """Determines the cache file path for a given benchmark name."""
    safe_bench_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", benchmark_name.lower())
    filename = f"cache_{safe_bench_name}.json"
    return cache_dir / filename

def load_cache(cache_file, cache_ttl_seconds):
    """Loads results from a cache file if it exists, is valid, and not expired."""
    # Args: cache_file (pathlib.Path): Path to the cache file.
    #       cache_ttl_seconds (int): Max age of cache in seconds.
    # Returns: dict | None: Loaded results data, or None if invalid/expired/missing.
    if not cache_file.is_file():
        return None # No cache file exists

    try:
        # Check cache expiration using the passed TTL
        mtime = cache_file.stat().st_mtime
        age_seconds = time.time() - mtime
        if age_seconds > cache_ttl_seconds:
            print(f"[INFO] Cache file {cache_file.name} is expired ({age_seconds/3600:.1f} hours old, TTL: {cache_ttl_seconds/3600:.1f}h). Ignoring.")
            # Consider deleting expired cache?
            # try: cache_file.unlink() except OSError: pass
            return None

        # Load and validate data
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Basic validation
        if not isinstance(data, dict):
            print(f"[WARN] Cache file {cache_file.name} has invalid format (not a dict). Ignoring.")
            return None
        has_summary = any(isinstance(v, dict) and "_summary" in v for k, v in data.items() if not k.startswith("_"))
        if not has_summary and len(data) > 3:
             print(f"[WARN] Cache file {cache_file.name} seems incomplete or invalid (missing summaries). Ignoring.")
             return None

        print(f"[INFO] Loaded valid cached results from {cache_file.name} (Created: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')})")
        return data

    except json.JSONDecodeError:
        print(f"[WARN] Cache file {cache_file.name} is corrupted (JSONDecodeError). Ignoring.")
        return None
    except Exception as e:
        print(f"[WARN] Failed to load or validate cache {cache_file.name}: {e}")
        return None

def save_cache(cache_file, data):
    """Saves the benchmark results data to a cache file."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[INFO] Benchmark results saved to cache: {cache_file}")
    except Exception as e:
        print(f"[WARN] Failed to save results to cache {cache_file}: {e}")

def clear_cache(cache_file):
    """Deletes the specified cache file if it exists."""
    if cache_file.is_file():
        try:
            print(f"[INFO] Deleting cache file: {cache_file}")
            cache_file.unlink()
        except OSError as e:
            print(f"[WARN] Could not delete cache file {cache_file}: {e}")
    else:
        print(f"[INFO] --clear-cache specified, but no cache file found at {cache_file}")