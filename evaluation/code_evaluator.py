# --- Code Execution and Testing ---

import re
import io
import contextlib
import traceback
import time
# Remove direct import of the constant
# from config import CODE_EXECUTION_TIMEOUT_SECONDS
from utils import (
    TimeoutException, setup_timeout_alarm, setup_timeout_fallback,
    cancel_timeout_alarm, cancel_timeout_fallback, check_timed_out,
    clear_timeout_flag, HAS_SIGNAL_ALARM, truncate_text
)

# Add code_execution_timeout parameter
def execute_and_test_code(response_text, function_name, test_cases, code_execution_timeout):
    """Safely executes model-generated code and runs test cases against it."""
    # Args: response_text (str): Code provided by the model.
    #       function_name (str): Expected function name.
    #       test_cases (list): Test cases with input/output.
    #       code_execution_timeout (int): Timeout in seconds per test case execution.
    # Returns: tuple(bool, str): (all_tests_passed, details_message)
    if not response_text.strip():
        return False, "No code provided by model"

    # Extract code block (Python assumed)
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    code_to_exec = code_match.group(1).strip() if code_match else response_text.strip()

    if not code_to_exec:
        return False, "Extracted code block is empty or response was only fences"

    # Prepare safe execution environment
    scope = {}
    safe_builtins = {
        "print": print, "range": range, "len": len, "abs": abs, "round": round,
        "str": str, "int": int, "float": float, "bool": bool, "list": list,
        "dict": dict, "tuple": tuple, "set": set, "max": max, "min": min, "pow": pow,
        "sum": sum, "sorted": sorted, "isinstance": isinstance, "any": any, "all": all,
        "True": True, "False": False, "None": None,
        "ValueError": ValueError, "TypeError": TypeError, "IndexError": IndexError,
        "KeyError": KeyError, "NameError": NameError, "Exception": Exception,
    }

    overall_success = True
    results_details = []
    definition_error = None
    timer = None # For fallback timeout

    # Calculate total timeout for definition + all test cases using the passed value
    total_timeout = code_execution_timeout * (len(test_cases) + 1)
    clear_timeout_flag() # Ensure flag is clear initially

    try:
        # --- Setup Timeout ---
        if HAS_SIGNAL_ALARM:
            setup_timeout_alarm(int(total_timeout))
        else:
            timer = setup_timeout_fallback(total_timeout)

        # --- Execute Code Definition ---
        exec_stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(exec_stdout): # Capture print statements during definition
                exec(code_to_exec, {"__builtins__": safe_builtins}, scope)
            if check_timed_out(): raise TimeoutException("Timeout during code definition (fallback)")
        except TimeoutException as te:
            raise te # Re-raise to be caught by outer handler
        except Exception as e:
            definition_error = f"Code definition error: {type(e).__name__}: {e}"
            overall_success = False
            # Fall through

        if check_timed_out(): raise TimeoutException("Timeout detected after code definition (check)")

        # --- Validate Function ---
        if function_name not in scope:
            err_suffix = f" ({definition_error})" if definition_error else ""
            raise NameError(f"Function '{function_name}' not defined.{err_suffix}")
        target_func = scope[function_name]
        if not callable(target_func):
            err_suffix = f" ({definition_error})" if definition_error else ""
            raise TypeError(f"'{function_name}' is defined but not callable.{err_suffix}")

        if definition_error:
            raise Exception(definition_error) # Caught by outer handler

        # --- Run Test Cases ---
        for idx, case in enumerate(test_cases):
            if check_timed_out(): raise TimeoutException("Timeout before starting test case (check)")

            inp = case.get("input", [])
            expected = case.get("expected_output")
            call_desc = f"{function_name}({', '.join(truncate_text(repr(x), 30) for x in inp)})"
            case_result_str = f"Case {idx+1}: {call_desc}"
            case_passed = False
            error_in_case = None

            case_stdout = io.StringIO()
            start_case_time = time.time()
            try:
                # Note: Individual case timeout isn't explicitly enforced here,
                # only the total_timeout for the whole block.
                # A more robust implementation might set a shorter timer per case.
                with contextlib.redirect_stdout(case_stdout):
                    actual_output = target_func(*inp)
                case_duration = time.time() - start_case_time

                if check_timed_out(): raise TimeoutException(f"Timeout detected after function call (check)")

                case_passed = compare_outputs(actual_output, expected)
                pass_fail_marker = "PASS" if case_passed else "FAIL"
                case_result_str += f" -> Got: {truncate_text(repr(actual_output), 50)}, Exp: {truncate_text(repr(expected), 50)} [{pass_fail_marker}]"

            except TimeoutException as te:
                error_in_case = f"TIMEOUT ({te})"
            except Exception as e:
                error_in_case = f"ERROR {type(e).__name__}: {truncate_text(str(e), 100)}"

            if error_in_case:
                case_result_str += f" -> {error_in_case}"
                overall_success = False
                case_passed = False
            elif not case_passed:
                 overall_success = False

            results_details.append(case_result_str)

    except TimeoutException as te:
        definition_error = f"Timeout Error: {te}"
        overall_success = False
    except (SyntaxError, NameError, TypeError) as e:
        definition_error = f"Definition/Setup Error: {type(e).__name__}: {e}"
        overall_success = False
    except Exception as e:
        definition_error = f"Unexpected Setup Error: {e}\n{traceback.format_exc(limit=2)}"
        overall_success = False
    finally:
        # --- Cleanup Timeout ---
        if HAS_SIGNAL_ALARM:
             cancel_timeout_alarm()
        if timer:
             cancel_timeout_fallback(timer)
        clear_timeout_flag()

    # --- Format Final Result ---
    final_details = "; ".join(results_details)
    if definition_error:
        final_details = f"{definition_error}. Test Cases: {final_details}" if final_details else definition_error

    final_details = truncate_text(final_details, 500)

    return overall_success, final_details


def compare_outputs(got, expected):
    """Compares function outputs with tolerance for numerics and basic types."""
    if type(got) != type(expected):
        if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
            return abs(float(got) - float(expected)) < 1e-9
        return False

    if isinstance(got, float):
        return abs(got - expected) < 1e-9

    if isinstance(got, (list, tuple)):
        if len(got) != len(expected): return False
        return all(compare_outputs(g, e) for g, e in zip(got, expected))

    if isinstance(got, dict):
        if set(got.keys()) != set(expected.keys()): return False
        return all(compare_outputs(got[k], expected[k]) for k in got)

    try:
        return got == expected
    except Exception as e:
        print(f"[WARN] Could not compare outputs ({type(got)}, {type(expected)}): {e}")
        return False