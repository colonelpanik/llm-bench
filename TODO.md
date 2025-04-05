### Version 0.2.0 (Completed)
- [X] **Multiple Valid Answers:** Logic recognizes multiple correct answers (e.g., 'any' match for keywords).
- [X] **Safer Code Execution:** Checks function scope and callability; handles basic errors.
- [X] **Non-UNIX Timeout Fallback:** Uses `threading.Timer` fallback for code execution timeout.
- [X] **Cognitive Load JSON Tolerance:** `compare_structured_objects` handles key order differences.

---

### Version 0.3.0 (Completed)
- [X] **Weighted Summaries & Explanations:** `partial_keyword_scoring` supports `weighted_keywords`.
- [X] **Enhanced Regex Validation:** `evaluate_with_regex` uses `regex_validation_rules`.
- [X] **Optional Confidence/Probability Checks:** `classification_confidence` type implemented.

---

### Version 0.4.0 (Completed)
- [X] **Aggregate Metrics & Weighted Final Score:** Category-weighted "Overall Weighted Score" implemented.
- [X] **Semantic Similarity for Summaries:** `evaluation_method: "semantic"` option implemented.
- [X] **Detailed Failure Logs:** Evaluation functions return detailed failure strings captured in results.

---

### Version 0.5.0 (Completed)
- [X] **Retry/ Rerun Tasks on Error:** Retry logic for transient API errors implemented in clients.
- [X] **Extended Memory Monitoring:** NVIDIA GPU memory tracking via `pynvml` added.
- [X] **Log Full Prompt & Response (Optional):** Full response stored in results; `--verbose` for console.

---

### Version 0.6.0 (Completed)
- [X] **Codebase Modularization:** Codebase refactored into multiple modules/packages.
- [X] **Improve HTML Report UI/UX:** Template enhancements for structure, style, definitions, plots. Visualization logic fixed (v0.7.0).
- [X] **Configuration File:** `config.yaml` implemented for managing defaults.

---

### Version 0.7.0 (Current - Focus of this Update)
- [X] **Fix Visualization Bug:** Corrected `group_by_stage` plotting logic in HTML report to use color-coding instead of broken bar repositioning. Plots now correctly show stage comparison within the ranked list.
- [X] **Implement Dependency Check Utility:** Added `--check-dependencies` CLI flag to verify optional libraries and their basic functions (psutil, pynvml, matplotlib, sentence-transformers, yaml).
- [X] **Implement CSV/JSON Export:** Added `--export-summary-csv` and `--export-details-json` CLI flags to export results to the report directory.

---

### Future Enhancements (Post V0.7.0)
- [ ] **Enhanced Testing:** Add more specific unit/integration tests for evaluation edge cases (empty responses, malformed data, timeouts) and CLI argument interactions.
- [ ] **Task Selection Enhancements:**
    - [ ] Add `--task-tag mytag` support (requires defining tags in `benchmark_tasks.json`).
    - [ ] Add exclusion filters like `--exclude-model llama3:8b` or `--exclude-task 'Sentiment.*'`.
- [ ] **Advanced Resource Monitoring:** Implement optional *in-task* polling (e.g., every 0.5s in a separate thread) for RAM/GPU to capture peak usage *during* generation, not just before/after. (Complexity: High)
- [ ] **Function Calling Evaluation:** Add a new task type (`function_call`) to evaluate LLM function/tool calling capabilities. Requires defining expected calls/arguments. (Complexity: High)
- [ ] **Progress Indicator:** Implement a more informative progress indicator using `tqdm` in `benchmark_runner.py` for model and task loops. (Complexity: Low-Medium)
- [ ] **Historical Run Comparison (Report):** Enhance the HTML report to optionally load data from a previous `cache_*.json` file (`--compare-run path/to/cache.json`) and display comparison tables/charts. (Complexity: Medium-High)
- [ ] **Async API Queries:** Refactor `benchmark_runner` to optionally use `asyncio` and `aiohttp` (or similar) to query multiple models concurrently. Add a `--concurrent-models N` flag. (Complexity: High)
- [ ] **Summarization Length Metric:** Add optional configuration in summarization tasks (`target_word_count`, `length_penalty_factor`) to calculate an additional *metric* (not just details string) based on summary length. (Complexity: Medium - requires changes to evaluation return signature/results structure).
- [ ] **Report Customization:** Allow choosing which plots or sections to include in the HTML report via CLI flags or `config.yaml`. (Complexity: Medium)