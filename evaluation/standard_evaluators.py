# --- Standard Evaluation Methods (Keywords, Regex, Confidence) ---

import re
import json
from utils import truncate_text

# --- Regex Evaluation ---

def evaluate_with_regex(task, response_text):
    """Checks regex patterns and optionally validates captures."""
    # Args: task (dict): Task definition including 'expected_regex_map' and optional 'regex_validation_rules'. response_text (str): Model output.
    # Returns: tuple(bool, str): (success_flag, details_message)
    regex_map = task.get("expected_regex_map", {})
    validation_rules = task.get("regex_validation_rules", {})
    if not regex_map:
        return False, "[CONFIG ERROR] Task missing 'expected_regex_map'"

    missing_patterns = []
    captures = {}
    validation_errors = []
    found_any_match = False # Track if at least one pattern matched

    for key, pattern in regex_map.items():
        try:
            # Use common flags for flexibility
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if not match:
                missing_patterns.append(key)
            else:
                found_any_match = True
                # Prefer group(1) if exists, else full match
                captured_value = match.group(1).strip() if len(match.groups()) > 0 and match.group(1) is not None else match.group(0).strip()
                captures[key] = captured_value

                # Apply validation if rules exist
                if key in validation_rules:
                    rule = validation_rules[key]
                    valid, error_msg = validate_capture(captured_value, rule)
                    if not valid:
                        validation_errors.append(f"Validation fail for '{key}' ('{truncate_text(captured_value, 50)}'): {error_msg}")

        except re.error as e:
            return False, f"[CONFIG ERROR] Invalid regex pattern for key '{key}': {e}"
        except IndexError:
             # Should be rare with the group check above, but handle defensively
             captures[key] = match.group(0).strip() # Fallback to full match

    # Determine final result
    if not found_any_match:
         return False, f"No expected regex patterns matched. Snippet: '{truncate_text(response_text)}'"
    elif missing_patterns:
        return False, f"Missing regex patterns for keys: {missing_patterns}. Captures so far: {captures}"
    elif validation_errors:
        return False, f"Regex patterns matched, but validation failed: {'; '.join(validation_errors)}. Captures: {captures}"
    else:
        return True, f"Matched all regex patterns. Captures: {captures}"

def validate_capture(value, rule):
    """Applies a single validation rule to a captured string."""
    # Args: value (str): The captured string. rule (dict): The validation rule definition.
    # Returns: tuple(bool, str): (is_valid, error_message)
    rule_type = rule.get("type")
    if not value and rule_type != "optional": # Empty capture fails unless explicitly optional
         return False, "Captured value is empty"

    try:
        if rule_type == "numeric":
            num_value = float(value.replace(',', '')) # Handle commas
            min_val = rule.get("min")
            max_val = rule.get("max")
            if min_val is not None and num_value < min_val: return False, f"Value {num_value} < min {min_val}"
            if max_val is not None and num_value > max_val: return False, f"Value {num_value} > max {max_val}"
            return True, ""
        elif rule_type == "string_contains":
            substring = rule.get("value")
            if substring is None: return False, "Missing 'value' for string_contains"
            if substring not in value: return False, f"Does not contain '{substring}'"
            return True, ""
        elif rule_type == "string_enum":
            allowed = rule.get("values")
            case_sensitive = rule.get("case_sensitive", False)
            if not isinstance(allowed, list): return False, "Missing/invalid 'values' list for enum"
            check_val = value if case_sensitive else value.lower()
            allowed_set = set(v if case_sensitive else v.lower() for v in allowed)
            if check_val not in allowed_set: return False, f"Not in allowed enum: {allowed}"
            return True, ""
        elif rule_type == "optional":
            return True, "" # Always valid if optional
        # Add more types: date, specific pattern match, etc.
        else:
            return False, f"Unknown validation rule type: '{rule_type}'"
    except ValueError:
        return False, f"Conversion failed for rule '{rule_type}' (e.g., not numeric)"
    except Exception as e:
        return False, f"Error during validation: {e}"


# --- Partial / Weighted Keyword Scoring ---

def partial_keyword_scoring(task, response_text):
    """Scores based on keyword presence, supports weighted keywords."""
    # Args: task (dict): Task definition with 'weighted_keywords' or 'expected_keywords'. response_text (str): Model output.
    # Returns: tuple(float, str): (score_percentage, details_message)
    weighted_kws = task.get("weighted_keywords")
    expected_kws = task.get("expected_keywords") # Fallback
    task_name = task.get("name", "Unnamed Task")

    if not weighted_kws and not expected_kws:
        return 0.0, "[CONFIG ERROR] No weighted or expected keywords defined for scoring"

    resp_lower = response_text.lower()
    achieved_score = 0.0
    total_possible_score = 0.0
    found_list = []
    missing_list = []
    items_to_check = []

    # Prepare items list with keywords and weights
    if weighted_kws:
        if not isinstance(weighted_kws, list): return 0.0, f"[CONFIG ERROR] Task '{task_name}': 'weighted_keywords' must be a list"
        for item in weighted_kws:
            if isinstance(item, dict) and "keyword" in item:
                kw = item.get("keyword")
                w = item.get("weight", 1.0)
                if isinstance(kw, str) and kw.strip() and isinstance(w, (int, float)) and w > 0:
                    items_to_check.append({"keyword": kw.strip(), "weight": float(w)})
                else: print(f"[WARN] Task '{task_name}': Invalid item in weighted_keywords: {item}")
            else: print(f"[WARN] Task '{task_name}': Invalid item format in weighted_keywords: {item}")
    elif expected_kws: # Fallback to unweighted
        if not isinstance(expected_kws, list): return 0.0, f"[CONFIG ERROR] Task '{task_name}': 'expected_keywords' must be a list"
        items_to_check = [{"keyword": kw.strip(), "weight": 1.0} for kw in expected_kws if isinstance(kw, str) and kw.strip()]

    if not items_to_check: return 0.0, f"[CONFIG ERROR] Task '{task_name}': No valid keywords found to score against"

    # Perform checks using word boundaries
    for item in items_to_check:
        keyword = item["keyword"]
        weight = item["weight"]
        total_possible_score += weight
        # Use word boundary regex for accuracy (?<!\w)keyword(?!\w)
        pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
        if re.search(pattern, resp_lower):
            achieved_score += weight
            found_list.append(f"{keyword}({weight:.1f})")
        else:
            missing_list.append(f"{keyword}({weight:.1f})")

    score_percent = (achieved_score / total_possible_score) * 100 if total_possible_score > 0 else 0.0

    # Format detail message
    detail = f"Score={score_percent:.1f}% ({achieved_score:.1f}/{total_possible_score:.1f} pts)."
    max_display = 5 # Limit displayed keywords
    if found_list: detail += f" Found: [{', '.join(found_list[:max_display])}{'...' if len(found_list) > max_display else ''}]."
    if missing_list: detail += f" Missing: [{', '.join(missing_list[:max_display])}{'...' if len(missing_list) > max_display else ''}]."

    return float(score_percent), detail


# --- Classification with Confidence ---

def evaluate_with_confidence(task, response_text, default_min_confidence):
    """Evaluates classification tasks requiring a specific label and minimum confidence score from JSON."""
    # Args: task (dict): Task definition. response_text (str): Model JSON output. default_min_confidence (float): Default threshold if not in task.
    # Returns: tuple(bool, str): (success_flag, details_message)
    label_key = task.get("expected_label_key")
    confidence_key = task.get("expected_confidence_key")
    expected_label_list = task.get("expected_keywords") # Should contain one expected label
    min_threshold = task.get("min_confidence_threshold", default_min_confidence)
    task_name = task.get("name", "Unnamed Task")

    # Validate task setup
    if not label_key or not confidence_key: return False, f"[CONFIG ERROR] Task '{task_name}': Missing 'expected_label_key' or 'expected_confidence_key'"
    if not expected_label_list or not isinstance(expected_label_list, list) or len(expected_label_list) != 1: return False, f"[CONFIG ERROR] Task '{task_name}': 'expected_keywords' must contain exactly one label string"
    if not isinstance(min_threshold, (float, int)) or not (0 <= min_threshold <= 1):
         print(f"[WARN] Task '{task_name}': Invalid min_confidence_threshold '{min_threshold}'. Using default {default_min_confidence}.")
         min_threshold = default_min_confidence
    expected_label = expected_label_list[0].strip()

    # Parse response JSON
    try:
        clean_response = re.sub(r'^$', '', clean_response).strip()
        if not clean_response: return False, "Response empty after cleaning JSON fences."
        data = json.loads(clean_response)
        if not isinstance(data, dict): return False, f"Parsed JSON is not a dictionary (got {type(data).__name__})."
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}. Snippet: '{truncate_text(response_text)}'"

    # Extract values
    actual_label_raw = data.get(label_key)
    actual_confidence_raw = data.get(confidence_key)

    if actual_label_raw is None: return False, f"Label key '{label_key}' not found in response JSON keys: {list(data.keys())}."
    if actual_confidence_raw is None: return False, f"Confidence key '{confidence_key}' not found in response JSON keys: {list(data.keys())}."

    # Normalize and validate values
    actual_label = str(actual_label_raw).strip()
    try:
        if isinstance(actual_confidence_raw, str) and '%' in actual_confidence_raw:
            actual_confidence = float(actual_confidence_raw.replace('%','').strip()) / 100.0
        else:
             actual_confidence = float(actual_confidence_raw)
        if not (0 <= actual_confidence <= 1):
             return False, f"Confidence value {actual_confidence:.3f} is outside the valid range [0, 1]."
    except (ValueError, TypeError):
        return False, f"Confidence value '{actual_confidence_raw}' is not a valid number or percentage string."

    # Perform checks (case-insensitive label compare)
    label_match = (actual_label.lower() == expected_label.lower())
    confidence_met = (actual_confidence >= min_threshold)

    # Determine result and details
    detail_msg = f"Label: Got '{actual_label}', Expected '{expected_label}'. Confidence: Got {actual_confidence:.3f}, Threshold >= {min_threshold:.3f}."
    if label_match and confidence_met:
        return True, f"PASS. {detail_msg}"
    elif label_match and not confidence_met:
        return False, f"FAIL (Confidence too low). {detail_msg}"
    elif not label_match and confidence_met:
        return False, f"FAIL (Incorrect label). {detail_msg}"
    else: # Neither match
        return False, f"FAIL (Incorrect label & low confidence). {detail_msg}"


# --- Simple Keyword Matching ---

def simple_keyword_match(task, response_text):
    """Checks if *any* of the expected keywords are present (case-insensitive, word boundary)."""
    # Args: task (dict): Task definition with 'expected_keywords'. response_text (str): Model output.
    # Returns: tuple(bool, str): (success_flag, details_message)
    expected_keywords = task.get("expected_keywords", [])
    if not expected_keywords: return False, "[CONFIG ERROR] simple_keyword_match called without 'expected_keywords'."

    resp_norm = response_text.lower()
    found_any = []
    for kw in expected_keywords:
        pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
        if re.search(pattern, resp_norm):
            found_any.append(kw)

    if found_any:
        return True, f"Matched keyword(s): {found_any} (any of {expected_keywords} required)"
    else:
        return False, f"Expected any of {expected_keywords}, none found. Snippet: '{truncate_text(response_text)}'"

def simple_keyword_check_strict(task, response_text):
    """Checks if *all* expected keywords are present (case-insensitive, word boundary)."""
    # Args: task (dict): Task definition with 'expected_keywords'. response_text (str): Model output.
    # Returns: tuple(bool, str): (success_flag, details_message)
    expected_keywords = task.get("expected_keywords", [])
    if not expected_keywords: return False, "[CONFIG ERROR] simple_keyword_check_strict called without 'expected_keywords'."

    resp_norm = response_text.lower()
    missing = []
    for kw in expected_keywords:
         pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
         if not re.search(pattern, resp_norm):
            missing.append(kw)

    if not missing:
        return True, "All required keywords found"
    else:
        return False, f"Missing required keywords: {missing}. Snippet: '{truncate_text(response_text)}'"

