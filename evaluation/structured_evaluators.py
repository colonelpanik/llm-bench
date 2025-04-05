# --- Structured Data Evaluation (JSON, YAML) ---

import json
import re
from utils import truncate_text

# --- Main Entry Point ---

def evaluate_structured_output(task, response_text, pyyaml_available, yaml_module):
    """Evaluates JSON/YAML structure and optionally compares against expected structure."""
    # Args: task (dict): Task definition. response_text (str): Model output. pyyaml_available (bool): Flag if PyYAML is loaded. yaml_module: Imported yaml library.
    # Returns: tuple(bool, str): (success_flag, details_message)
    task_type = task.get("type", "unknown")
    is_yaml = task_type.startswith("yaml_")
    parser_name = "YAML" if is_yaml else "JSON"
    exp_struct = task.get("expected_structure")
    task_name = task.get("name", "Unnamed Task")

    # Check dependency for YAML
    if is_yaml and not pyyaml_available:
        return False, f"Cannot evaluate YAML task '{task_name}': PyYAML library not available."


    # Clean code fences - CORRECTED
    clean_response = response_text.strip() # Start with the raw stripped text
    clean_response = re.sub(r'^```[a-zA-Z]*(?:yaml|json)?\n?', '', clean_response, flags=re.IGNORECASE)
    clean_response = re.sub(r'\n?```$', '', clean_response).strip()

    if not clean_response:
         # If expected structure is None (meaning just valid parsing needed), empty is a fail.
         # If expected structure is defined, empty clearly doesn't match.
         return False, "Response is empty after cleaning code fences."

    # Parse the cleaned response
    parsed_obj = None
    parse_error = None
    try:
        if is_yaml:
            # Use safe_load to prevent arbitrary code execution
            parsed_obj = yaml_module.safe_load(clean_response)
        else:
            parsed_obj = json.loads(clean_response)
    except Exception as e: # Catch parsing errors
        if is_yaml and pyyaml_available: # Check if it was a YAMLError
             if hasattr(e, 'problem_mark'): # Nicer error for YAML parse issues
                 parse_error = f"{parser_name} parse error: {e} at line {e.problem_mark.line+1}, col {e.problem_mark.column+1}"
             else:
                  parse_error = f"{parser_name} parse error: {e}"
        else: # JSONDecodeError or other exception
            parse_error = f"{parser_name} parse error: {type(e).__name__}: {e}"

    if parse_error:
        return False, f"{parse_error}. Snippet: '{truncate_text(response_text)}'"

    # --- Evaluation Logic ---

    # Case 1: Parsing failed (handled above)

    # Case 2: Parsing succeeded, but resulted in None (e.g., empty YAML file)
    if parsed_obj is None:
        # Fail if structure was expected, or if non-empty input parsed to None
        if exp_struct is not None:
             return False, f"{parser_name} parsed to None, but expected structure was provided."
        elif clean_response.strip(): # Non-empty input resulted in None
             return False, f"{parser_name} parsed successfully but result is None from non-empty input."
        else: # Empty input parsed to None - technically valid parse, but maybe not desired? Treat as fail for now.
            return False, f"{parser_name} parsed successfully, but result is None (input likely empty)."


    # Case 3: Parsing succeeded, check against expected structure if provided
    if exp_struct is not None:
        match, detail = compare_structured_objects(exp_struct, parsed_obj)
        if match:
            return True, f"{parser_name} structure matches expected."
        else:
            # Provide details from the comparison function
            return False, f"{parser_name} mismatch: {detail}. Snippet: '{truncate_text(clean_response)}'"

    # Case 4: Parsing succeeded, no expected structure provided
    # If we reach here, parsed_obj is not None.
    return True, f"{parser_name} parsed successfully (no expected structure defined)."


# --- Comparison Logic ---

def compare_structured_objects(obj1, obj2):
    """Recursively compares two objects (dicts, lists, primitives) with tolerance."""
    # Args: obj1: The first object (expected). obj2: The second object (actual).
    # Returns: tuple(bool, str): (match_flag, mismatch_detail_string)
    type1 = type(obj1)
    type2 = type(obj2)

    # Type mismatch (allow int/float comparison)
    if type1 != type2:
        if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
            if abs(float(obj1) - float(obj2)) < 1e-9: # Use tolerance for float comparison
                return True, ""
            else:
                return False, f"Numeric value mismatch: expected {obj1}, got {obj2}"
        else:
            return False, f"Type mismatch: expected {type1.__name__}, got {type2.__name__}"

    # Dictionary comparison (order-insensitive keys)
    if isinstance(obj1, dict):
        keys1 = set(obj1.keys())
        keys2 = set(obj2.keys())
        if keys1 != keys2:
            missing = keys1 - keys2
            extra = keys2 - keys1
            details = []
            if missing: details.append(f"Missing keys: {missing}")
            if extra: details.append(f"Extra keys: {extra}")
            return False, "; ".join(details)
        # Recursively compare values for matching keys
        for k in obj1:
            match, detail = compare_structured_objects(obj1[k], obj2[k])
            if not match:
                return False, f"Mismatch at key '{k}': {detail}"
        return True, "" # All keys and values matched

    # List comparison (order-sensitive)
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False, f"List length mismatch: expected {len(obj1)}, got {len(obj2)}"
        # Recursively compare elements
        for i in range(len(obj1)):
            match, detail = compare_structured_objects(obj1[i], obj2[i])
            if not match:
                return False, f"Mismatch at list index {i}: {detail}"
        return True, "" # All elements matched

    # Primitive comparison (string, bool, None, already handled numeric types)
    else:
        if obj1 == obj2:
            return True, ""
        else:
            # Provide value context for primitives
            return False, f"Value mismatch: expected '{truncate_text(str(obj1))}', got '{truncate_text(str(obj2))}'"


def fuzzy_json_compare(expected_json_str, actual_response):
    """Legacy helper primarily for cognitive_load tasks (uses compare_structured_objects)."""
    # Args: expected_json_str (str): Expected JSON string. actual_response (str): Model's response string.
    # Returns: tuple(bool, str): (match_flag, details_message)
    try:
        clean_actual = re.sub(r'^$', '', clean_actual).strip()
        if not clean_actual: return False, "Empty response after cleaning code fences"

        expected_obj = json.loads(expected_json_str)
        actual_obj = json.loads(clean_actual)
    except json.JSONDecodeError as e:
        return False, f"JSON parse fail: {e} | Resp snippet: '{truncate_text(actual_response)}'"
    except Exception as e:
         return False, f"Error during JSON preparation: {e}"

    # Use the main comparison function
    match, detail = compare_structured_objects(expected_obj, actual_obj)
    if match:
        return True, "Fuzzy JSON match success (structure and values OK)"
    else:
        return False, f"JSON mismatch: {detail}. Snippet: '{truncate_text(clean_actual)}'"

