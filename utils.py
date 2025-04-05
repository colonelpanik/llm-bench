# --- Utility Functions and Timeout Handling ---

import time
import signal
import threading
import traceback
import re 

# --- Formatting ---
def format_na(value, suffix="", precision=1):
    """Helper to format numbers or return N/A string."""
    # Args: value: The number or None to format. suffix (str): String to append. precision (int): Decimal places for floats.
    # Returns: str: Formatted string or "N/A".
    if value is None or value != value: # Checks for None or NaN
        return "N/A"
    try:
        if isinstance(value, float):
            return f"{value:.{precision}f}{suffix}"
        elif isinstance(value, int):
            return f"{value}{suffix}"
        else:
            return f"{str(value)}{suffix}"
    except (ValueError, TypeError):
        return "N/A"

def truncate_text(text, max_len=150):
    """Truncate text to a maximum length for display."""
    # Args: text (str): The text to truncate. max_len (int): Maximum length.
    # Returns: str: Truncated text or original if shorter.
    if not text: return ""
    return (text[:max_len] + '...') if len(text) > max_len else text

# --- Timeout Handling for Code Execution ---
class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

_timed_out_flag = threading.Event() # Flag for fallback timer

def timeout_handler(signum, frame):
    """Signal handler for SIGALRM."""
    # Args: signum: Signal number. frame: Current stack frame.
    # Raises: TimeoutException
    raise TimeoutException("Code execution timed out via SIGALRM")

def fallback_timeout_handler(process_event):
    """Sets the event flag when the threading.Timer expires."""
    # Args: process_event (threading.Event): The event to set.
    # Returns: None
    print("[WARN] Code execution fallback timer expired.")
    traceback.print_stack() # Print stack to help debug where timeout occurred
    process_event.set()

# Check if SIGALRM is available (not on Windows)
HAS_SIGNAL_ALARM = hasattr(signal, "SIGALRM")

def check_timed_out():
    """Checks if the fallback timeout flag is set."""
    # Returns: bool: True if the timeout flag is set, False otherwise.
    return _timed_out_flag.is_set()

def clear_timeout_flag():
    """Clears the fallback timeout flag."""
    # Returns: None
    _timed_out_flag.clear()

def setup_timeout_alarm(duration_seconds):
    """Sets up the SIGALRM timeout."""
    # Args: duration_seconds (int): Timeout duration.
    # Returns: None
    if HAS_SIGNAL_ALARM:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration_seconds)

def setup_timeout_fallback(duration_seconds):
    """Sets up the threading.Timer fallback timeout."""
    # Args: duration_seconds (float): Timeout duration.
    # Returns: threading.Timer: The timer object.
    timer = threading.Timer(duration_seconds, fallback_timeout_handler, args=[_timed_out_flag])
    timer.daemon = True # Allow program to exit even if timer thread is running
    timer.start()
    return timer

def cancel_timeout_alarm():
    """Cancels the SIGALRM timeout."""
    # Returns: None
    if HAS_SIGNAL_ALARM:
        signal.alarm(0) # Disable the alarm
        signal.signal(signal.SIGALRM, signal.SIG_DFL) # Restore default handler

def cancel_timeout_fallback(timer):
    """Cancels the threading.Timer fallback."""
    # Args: timer (threading.Timer): The timer object to cancel.
    # Returns: None
    if timer:
        timer.cancel()

