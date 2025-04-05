# -*- coding: utf-8 -*-
"""
embed_bench.py

Benchmark script for comparing sentence embedding models on intent classification.
Includes speed (train & inference tokens/sec), accuracy, size metrics.
Outputs a summary table, a PNG graph, and an HTML report to a subdirectory.
"""

import json
import time
import os
import shutil # For directory removal
import numpy as np
import pandas as pd # For HTML table generation
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for saving plots without display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
import warnings

# --- Configuration ---
# Updated list of model names
# Updated list of model names - VERIFIED on Hugging Face
#MODEL_NAMES = [
#    "distilbert-base-uncased",                # Verified: Exists
#    "google/mobilebert-uncased",              # Verified: Exists
#    "sentence-transformers/all-MiniLM-L6-v2", # Verified: Exists
#    "google/electra-small-discriminator",     # Verified: Exists
#    "albert/albert-base-v2",                  # Verified: Exists
#    "Lajavaness/bilingual-embedding-large",   # Verified: Exists
#    "intfloat/multilingual-e5-large-instruct",# Verified: Exists
#    "jeffh/intfloat-multilingual-e5-large-instruct", # Added & Verified: Exists
#    "NovaSearch/jasper_en_vision_language_v1",  # Added & Verified: Exists (Vision-Language)
#    "HIT-TMG/KaLM-embedding-multilingual-mini-v1" # Added & Verified: Exists
#]

# Updated dictionary with approximate model sizes in MB (FP32) - VERIFIED names
#APPROX_MODEL_SIZES_MB = {
#    "distilbert-base-uncased": 268,
#    "google/mobilebert-uncased": 98,
##    "sentence-transformers/all-MiniLM-L6-v2": 86,
#    "google/electra-small-discriminator": 54,
#    "albert/albert-base-v2": 47,
#    "Lajavaness/bilingual-embedding-large": 1372,  # (~1.34 GB)
#    "intfloat/multilingual-e5-large-instruct": 2294, # (~2.24 GB)
#    "jeffh/intfloat-multilingual-e5-large-instruct": 2294, # Added (~2.24 GB)
#    "NovaSearch/jasper_en_vision_language_v1": 605,    # Added (~605 MB) - Note: Vision-Language Model
#    "HIT-TMG/KaLM-embedding-multilingual-mini-v1": 126 # Added (~126 MB)
#}
MODEL_NAMES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/distiluse-base-multilingual-cased-v1", # YAML mentioned v1 or v2
    "google/LaBSE"
]

# Approximate model sizes in MB (FP32) derived from the provided YAML
APPROX_MODEL_SIZES_MB = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 471,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 1110, # Approx 1.11 GB
    "intfloat/multilingual-e5-base": 1110, # Approx 1.11 GB
    "intfloat/multilingual-e5-small": 450,
    "sentence-transformers/distiluse-base-multilingual-cased-v1": 541, # YAML mentioned v1 or v2
    "google/LaBSE": 1880 # Approx 1.88 GB
}

# Input JSON data file name
INPUT_DATA_FILE = 'synthetic_benchmark_data.json'

# --- Output Configuration ---
# Subdirectory for storing results
OUTPUT_DIR = 'benchmark_reports'
# Timestamp for unique filenames
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
# Output plot and HTML filenames (will be placed in OUTPUT_DIR)
PLOT_BASENAME = f"benchmark_results_{TIMESTAMP}.png"
HTML_BASENAME = f"benchmark_report_{TIMESTAMP}.html"

# Device selection
DEVICE = 'cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu'
# DEVICE = 'cpu' # Force CPU

TRUST_REMOTE_CODE_MODELS = ['e5', 'mxbai']
EMBEDDING_BATCH_SIZE = 128

# --- Helper Functions (load_data, preprocess_data remain the same as previous version) ---

def load_data(filepath):
    """Loads training and testing data from JSON file."""
    print(f"Attempting to load data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {filepath}")
        if 'train_data' not in data or 'test_prompts' not in data:
             raise KeyError("JSON file structure error: Must contain top-level keys 'train_data' and 'test_prompts'.")
        if not isinstance(data['train_data'], list) or not isinstance(data['test_prompts'], list):
             raise TypeError("'train_data' and 'test_prompts' must be lists in the JSON file.")
        if not data['train_data']: print("WARNING: The 'train_data' list in the JSON file is empty.")
        if not data['test_prompts']: print("WARNING: The 'test_prompts' list in the JSON file is empty. No evaluation possible.")
        return data['train_data'], data['test_prompts']
    except FileNotFoundError:
        print(f"ERROR: Input data file not found at the specified path: {filepath}"); exit(1)
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not load or parse JSON file. Check format. Error: {e}"); exit(1)
    except Exception as e:
         print(f"ERROR: An unexpected error occurred during data loading: {e}"); exit(1)

def preprocess_data(train_data, test_prompts):
    """Extracts texts/labels, checks format, encodes labels, filters test data."""
    train_texts, train_labels_raw, invalid_train_count = [], [], 0
    print("Processing train_data...")
    for i, item in enumerate(tqdm(train_data, desc="Processing train data")):
        if (isinstance(item, dict) and 'text' in item and isinstance(item['text'], str) and item['text'].strip() and
            'intent' in item and isinstance(item['intent'], str) and item['intent'].strip()):
            train_texts.append(item['text'].strip()); train_labels_raw.append(item['intent'].strip())
        else:
            if invalid_train_count < 10: print(f"WARNING: Skipping invalid entry train_data[{i}]: {str(item)[:100]}...")
            elif invalid_train_count == 10: print("WARNING: Further train_data warnings suppressed.")
            invalid_train_count += 1
    if invalid_train_count > 0: print(f"WARNING: Skipped {invalid_train_count} invalid entries in train_data.")

    test_texts, test_labels_raw, invalid_test_count = [], [], 0
    print("Processing test_prompts...")
    for i, item in enumerate(tqdm(test_prompts, desc="Processing test data")):
         if (isinstance(item, dict) and 'text' in item and isinstance(item['text'], str) and item['text'].strip() and
             'true_intent' in item and isinstance(item['true_intent'], str) and item['true_intent'].strip()):
            test_texts.append(item['text'].strip()); test_labels_raw.append(item['true_intent'].strip())
         else:
            if invalid_test_count < 10: print(f"WARNING: Skipping invalid entry test_prompts[{i}]: {str(item)[:100]}...")
            elif invalid_test_count == 10: print("WARNING: Further test_prompts warnings suppressed.")
            invalid_test_count += 1
    if invalid_test_count > 0: print(f"WARNING: Skipped {invalid_test_count} invalid entries in test_prompts.")

    if not train_texts: print("ERROR: No valid training data found."); exit(1)
    if not test_texts: print("ERROR: No valid test data found."); exit(1)
    print(f"Using {len(train_texts)} valid training examples. Initially found {len(test_texts)} valid test prompts.")

    label_encoder = LabelEncoder()
    try:
        train_labels_encoded = label_encoder.fit_transform(train_labels_raw)
        print(f"Encoded {len(label_encoder.classes_)} unique intents from training data.")
    except Exception as e: print(f"ERROR during training label encoding: {e}"); exit(1)

    known_labels = set(label_encoder.classes_)
    valid_test_indices = [i for i, label in enumerate(test_labels_raw) if label in known_labels]
    if len(valid_test_indices) < len(test_texts):
        print(f"WARNING: {len(test_texts) - len(valid_test_indices)} test prompts excluded due to unknown labels.")
    test_texts_filtered = [test_texts[i] for i in valid_test_indices]
    test_labels_raw_filtered = [test_labels_raw[i] for i in valid_test_indices]
    if not test_texts_filtered: print("ERROR: No test prompts remaining after filtering."); exit(1)
    try: test_labels_encoded = label_encoder.transform(test_labels_raw_filtered)
    except Exception as e: print(f"ERROR during test label encoding: {e}"); exit(1)
    print(f"Using {len(test_texts_filtered)} test prompts for final evaluation.")
    return train_texts, train_labels_encoded, test_texts_filtered, test_labels_encoded, label_encoder


# --- Main Benchmarking Logic (run_benchmark remains the same as previous version) ---

def run_benchmark(model_name, train_texts, train_labels_encoded, test_texts, test_labels_encoded, device):
    """Runs the full benchmark (loading, embedding, training, eval) for a single model."""
    print(f"\n--- Benchmarking Model: {model_name} ---")
    results = {"model_name": model_name}
    model = None
    try:
        # 1. Load Model
        print(f"Loading model '{model_name}' onto device '{device}'...")
        start_load_time = time.time()
        trust_remote = any(frag in model_name.lower() for frag in TRUST_REMOTE_CODE_MODELS)
        print(f"Using trust_remote_code={trust_remote}")
        model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote)
        results["load_time_s"] = time.time() - start_load_time
        print(f"Model loaded successfully in {results['load_time_s']:.2f} seconds.")
        results["approx_size_mb"] = APPROX_MODEL_SIZES_MB.get(model_name, "Unknown")
        try: results["embedding_dim"] = model.get_sentence_embedding_dimension()
        except Exception as e: print(f"Warning: Could not get embedding dimension: {e}"); results["embedding_dim"] = "N/A"
        print(f"Approx Size: {results.get('approx_size_mb', 'N/A')} MB, Dim: {results.get('embedding_dim', 'N/A')}")

        # 2. Generate Embeddings for Training Data & Calculate Training Speed
        print("Generating embeddings for training data...")
        start_embed_train_time = time.time()
        train_embeddings = model.encode(train_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True, device=device, batch_size=EMBEDDING_BATCH_SIZE)
        train_embedding_duration = time.time() - start_embed_train_time
        results["train_embedding_time_s"] = train_embedding_duration
        print(f"Training embeddings generated in {train_embedding_duration:.2f} seconds.")
        total_tokens_train = 0
        train_tok_sec = 0
        if hasattr(model, 'tokenizer'):
            try:
                print("Tokenizing training set for speed calculation...")
                tokenizer = model.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                for text in tqdm(train_texts, desc="Tokenizing Train Set"):
                    encoding = tokenizer(text, return_tensors=None, add_special_tokens=True); total_tokens_train += len(encoding['input_ids'])
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
                print(f"Total tokens in training set (for {model_name}): {total_tokens_train:,}")
                if total_tokens_train > 0 and train_embedding_duration > 0: train_tok_sec = total_tokens_train / train_embedding_duration
                results["train_tokens_per_second"] = train_tok_sec
                print(f"Training embedding speed: {train_tok_sec:,.1f} tokens/second")
            except Exception as e: print(f"WARNING: Training tokenization failed: {e}"); results["train_tokens_per_second"] = -1
        else: print(f"WARNING: Cannot calc train tokens/sec for {model_name}."); results["train_tokens_per_second"] = -1

        # 3. Train Classifier
        print("Training classifier (Logistic Regression)...")
        start_train_time = time.time()
        classifier = LogisticRegression(max_iter=1500, random_state=42, solver='liblinear', C=1.0)
        classifier.fit(train_embeddings, train_labels_encoded)
        results["classifier_train_time_s"] = time.time() - start_train_time
        print(f"Classifier trained in {results['classifier_train_time_s']:.2f} seconds.")

        # 4. Generate Embeddings for Test Prompts & Measure Inference Speed (Tokens/Sec)
        print("Generating embeddings for test prompts and measuring inference speed...")
        total_tokens_test = 0
        infer_tok_sec = 0
        if hasattr(model, 'tokenizer'):
            try:
                print("Tokenizing test set for speed calculation...")
                tokenizer = model.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                for text in tqdm(test_texts, desc="Tokenizing Test Set"):
                    encoding = tokenizer(text, return_tensors=None, add_special_tokens=True); total_tokens_test += len(encoding['input_ids'])
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
                print(f"Total tokens in test set (for {model_name}): {total_tokens_test:,}")
            except Exception as e: print(f"WARNING: Test tokenization failed: {e}"); total_tokens_test = -1
        else: print(f"WARNING: Cannot calc inference tokens/sec for {model_name}."); total_tokens_test = -1

        start_embed_test_time = time.time()
        test_embeddings = model.encode(test_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True, device=device, batch_size=EMBEDDING_BATCH_SIZE)
        test_embedding_duration = time.time() - start_embed_test_time
        results["test_embedding_time_s"] = test_embedding_duration
        num_test_prompts = len(test_texts)
        results["avg_inference_time_ms"] = (test_embedding_duration / num_test_prompts * 1000) if num_test_prompts > 0 else 0
        if total_tokens_test > 0 and test_embedding_duration > 0: infer_tok_sec = total_tokens_test / test_embedding_duration
        results["inference_tokens_per_second"] = infer_tok_sec
        print(f"Test embeddings generated in {test_embedding_duration:.2f} seconds.")
        print(f"Average inference time per prompt: {results['avg_inference_time_ms']:.2f} ms")
        if total_tokens_test >= 0: print(f"Inference speed: {infer_tok_sec:,.1f} tokens/second")
        else: print("Inference speed: N/A")

        # 5. Predict Intents & Calculate Accuracy
        print("Predicting intents for test prompts...")
        predicted_labels_encoded = classifier.predict(test_embeddings)
        results["accuracy"] = accuracy_score(test_labels_encoded, predicted_labels_encoded)
        print(f"Accuracy on test prompts: {results['accuracy']:.4f}")

    except Exception as e:
        print(f"ERROR: Benchmark failed for model {model_name}: {e}")
        results['accuracy'] = 0.0; results['train_tokens_per_second'] = 0.0; results['inference_tokens_per_second'] = 0.0
        return None
    finally: # Cleanup
        if 'model' in locals() and model is not None: del model
        if 'train_embeddings' in locals(): del train_embeddings
        if 'test_embeddings' in locals(): del test_embeddings
        if 'classifier' in locals(): del classifier
        if device == 'cuda':
            try: import torch; torch.cuda.empty_cache()
            except ImportError: pass
            except Exception as e_cuda: print(f"Warning: Error clearing CUDA cache: {e_cuda}")
    return results

# --- Plotting Function (saves plot to file) ---
def plot_results(results_list, output_dir, plot_basename):
    """Generates and saves bar charts to a file within the output directory."""
    if not results_list: print("No valid results available to plot."); return None
    plot_filepath = os.path.join(output_dir, plot_basename)
    print(f"\nGenerating results plot: {plot_filepath}")
    valid_results = [r for r in results_list if r is not None]; valid_results.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
    if not valid_results: print("No valid results found after filtering. Cannot generate plot."); return None

    models = [r.get('model_name', 'Unknown').split('/')[-1] for r in valid_results]
    accuracies = [r.get('accuracy', 0) for r in valid_results]
    inference_tokens_sec = [max(0, r.get('inference_tokens_per_second', 0)) for r in valid_results]
    sizes = [r.get('approx_size_mb', 0) if isinstance(r.get('approx_size_mb'), (int, float)) else 0 for r in valid_results]
    num_models = len(models)
    if num_models == 0: print("No models with valid results to plot."); return None

    fig_width = max(8, num_models * 1.5 + 1)
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except OSError: print("Warning: Plot style not found. Using default.")

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 10), sharex=True)
    fig.suptitle('Embedding Model Benchmark Results', fontsize=16, y=0.98)
    ax1, ax3 = axes[0], axes[1]; ax2 = ax1.twinx() # Define axes clearly
    bar_width = 0.35; x = np.arange(num_models)

    bars1 = ax1.bar(x - bar_width/2, accuracies, bar_width, label='Accuracy', color='cornflowerblue', zorder=3)
    ax1.set_ylabel('Accuracy', color='cornflowerblue', fontsize=12, weight='bold')
    ax1.tick_params(axis='y', labelcolor='cornflowerblue', labelsize=10)
    ax1.set_ylim(0, max(1.0, max(accuracies) * 1.1) if accuracies else 1.0)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    bars2 = ax2.bar(x + bar_width/2, inference_tokens_sec, bar_width, label='Inference Tok/Sec', color='lightcoral', zorder=3)
    ax2.set_ylabel('Inference Tokens/Sec', color='lightcoral', fontsize=12, weight='bold')
    ax2.tick_params(axis='y', labelcolor='lightcoral', labelsize=10)
    ax2.set_ylim(0, max(inference_tokens_sec) * 1.15 if inference_tokens_sec and max(inference_tokens_sec) > 0 else 1)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    ax1.bar_label(bars1, fmt='{:.3f}', padding=3, fontsize=9, weight='medium')
    ax2.bar_label(bars2, fmt='{:,.0f}', padding=3, fontsize=9, weight='medium')
    ax1.set_title('Accuracy vs. Inference Speed', fontsize=14, pad=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
    legend_y_offset = -0.15 - (0.03 * (num_models // 4)); ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, legend_y_offset), ncol=2, fontsize=10)

    bars3 = ax3.bar(x, sizes, color='mediumseagreen', zorder=3, width=bar_width*1.5)
    ax3.set_ylabel('Approx. Model Size (MB)', fontsize=12, weight='bold')
    ax3.set_xlabel('Model Name (Sorted by Accuracy)', fontsize=12, weight='bold')
    ax3.tick_params(axis='x', labelsize=10); ax3.set_xticks(x); ax3.set_xticklabels(models, rotation=15, ha='right')
    ax3.set_ylim(0, max(sizes) * 1.15 if sizes and max(sizes) > 0 else 1)
    ax3.bar_label(bars3, fmt='{:,.0f} MB', padding=3, fontsize=9, weight='medium')
    ax3.set_title('Approximate Model Size', fontsize=14, pad=15)
    ax3.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    try:
        plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
        print(f"Benchmark results graph saved successfully to: {plot_filepath}")
        plt.close(fig) # Close plot figure after saving
        return plot_filepath # Return the path for embedding in HTML
    except Exception as e:
        print(f"ERROR: Failed to save plot file '{plot_filepath}'. Error: {e}")
        plt.close(fig)
        return None

# --- HTML Report Generation ---
# --- HTML Report Generation ---
def generate_html_report(results_list, plot_filepath, html_filepath):
    """Generates an HTML report with summary table and embedded plot."""
    if not results_list:
        print("No results to generate HTML report.")
        return

    print(f"Generating HTML report: {html_filepath}")
    valid_results = [r for r in results_list if r is not None]
    if not valid_results:
         print("No valid results found after filtering for HTML report.")
         return

    # Create DataFrame for better formatting
    df = pd.DataFrame(valid_results)

    # Define desired column order and rename for presentation
    column_map = {
        'model_name': 'Model Name',
        'approx_size_mb': 'Size (MB)',
        'embedding_dim': 'Dim',
        'accuracy': 'Accuracy',
        'train_tokens_per_second': 'Train Tok/s',
        'inference_tokens_per_second': 'Infer Tok/s',
        'avg_inference_time_ms': 'Inf ms/q'
    }
    # Filter DataFrame to only include columns we want in the table
    # Make sure all keys exist in the results before trying to access
    display_columns = [k for k in column_map.keys() if k in df.columns]
    df_display = df[display_columns].copy()
    df_display.rename(columns=column_map, inplace=True)

    # --- Corrected Formatters ---
    # Ensure formatters ALWAYS return strings
    float_format = "{:,.2f}".format
    int_format = "{:,.0f}".format
    percent_format = "{:.4f}".format

    formatters = {
        # Explicitly return 'N/A' or str(x) in else cases
        'Size (MB)': lambda x: int_format(x) if pd.notnull(x) and isinstance(x, (int, float, np.number)) else ('N/A' if pd.isnull(x) else str(x)),
        'Dim': lambda x: int_format(x) if pd.notnull(x) and isinstance(x, (int, float, np.number)) else ('N/A' if pd.isnull(x) else str(x)),
        'Accuracy': lambda x: percent_format(x) if pd.notnull(x) and isinstance(x, (float, np.number)) else ('N/A' if pd.isnull(x) else str(x)),
        'Train Tok/s': lambda x: int_format(x) if pd.notnull(x) and isinstance(x, (int, float, np.number)) and x > 0 else ('N/A' if pd.notnull(x) and x < 0 else '0'), # Handle 0, negative (N/A), and null
        'Infer Tok/s': lambda x: int_format(x) if pd.notnull(x) and isinstance(x, (int, float, np.number)) and x > 0 else ('N/A' if pd.notnull(x) and x < 0 else '0'), # Handle 0, negative (N/A), and null
        'Inf ms/q': lambda x: float_format(x) if pd.notnull(x) and isinstance(x, (float, np.number)) else ('N/A' if pd.isnull(x) else str(x))
    }
    # --- End Corrected Formatters ---


    # Convert DataFrame to HTML table with formatting and styling
    try:
        html_table = df_display.to_html(
            index=False,
            escape=False,
            formatters=formatters,
            border=0,
            justify='center', # Alignment requires string lengths, hence the need for robust formatters
            classes='benchmark-table'
        )
    except Exception as e_html:
        print(f"ERROR: Failed during DataFrame to HTML conversion: {e_html}")
        print("Attempting HTML generation without formatting...")
        # Fallback: Try generating without formatters if they cause issues
        try:
             html_table = df_display.to_html(index=False, escape=False, border=0, justify='center', classes='benchmark-table')
        except Exception as e_html_fallback:
             print(f"ERROR: Fallback HTML generation also failed: {e_html_fallback}")
             html_table = "<p>Error generating results table.</p>"


    # --- Construct Full HTML Page ---
    html_css = """
<style>
    body { font-family: sans-serif; margin: 20px; }
    h1, h2 { color: #333; }
    .benchmark-table {
        border-collapse: collapse;
        width: 90%;
        margin: 20px auto;
        border: 1px solid #ddd;
        font-size: 0.9em;
    }
    .benchmark-table th, .benchmark-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .benchmark-table th {
        background-color: #f2f2f2;
        font-weight: bold;
        padding-top: 12px;
        padding-bottom: 12px;
    }
    .benchmark-table tr:nth-child(even) { background-color: #f9f9f9; }
    .benchmark-table tr:hover { background-color: #e2e2e2; }
    .plot-container { text-align: center; margin-top: 30px; }
    img { max-width: 100%; height: auto; border: 1px solid #ccc; }
    p { color: #555; font-size: 0.9em; }
</style>
"""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Model Benchmark Report</title>
    {html_css}
</head>
<body>
    <h1>Embedding Model Benchmark Report</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Device Used: {DEVICE.upper()}</p>
    <p>Input Data: {INPUT_DATA_FILE}</p>

    <h2>Summary Table</h2>
    <p>Models sorted by Inference Tokens/Second (Descending), then Accuracy (Descending).</p>
    {html_table}

    <h2>Performance Graphs</h2>
    <div class="plot-container">
"""
    if plot_filepath and os.path.exists(plot_filepath):
         plot_basename_for_html = os.path.basename(plot_filepath)
         html_content += f'        <img src="{plot_basename_for_html}" alt="Benchmark Results Plot">\n'
         html_content += '        <p>Graph shows Accuracy vs Inference Speed (Top) and Model Size (Bottom).</p>\n'
    else:
         html_content += '        <p>Benchmark results plot could not be generated or found.</p>\n'

    html_content += """
    </div>
</body>
</html>
"""

    # Write HTML content to file
    try:
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report saved successfully to: {html_filepath}")
    except Exception as e:
        print(f"ERROR: Failed to write HTML report file '{html_filepath}'. Error: {e}")

# --- Script Execution ---
if __name__ == "__main__":
    print("Starting Benchmark Script...")
    print(f"Using Device: {DEVICE.upper()}")
    print(f"Models to benchmark: {MODEL_NAMES}")
    start_total_time = time.time()

    # --- Prepare Output Directory ---
    print(f"Ensuring clean output directory: '{OUTPUT_DIR}'")
    try:
        if os.path.exists(OUTPUT_DIR):
            # Remove existing directory and all its contents
            shutil.rmtree(OUTPUT_DIR)
            print(f"Removed previous contents of '{OUTPUT_DIR}'.")
        # Create the directory
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory '{OUTPUT_DIR}'.")
    except Exception as e:
        print(f"ERROR: Could not prepare output directory '{OUTPUT_DIR}'. Check permissions. Error: {e}")
        exit(1)

    # Construct full paths for output files
    plot_filepath_full = os.path.join(OUTPUT_DIR, PLOT_BASENAME)
    html_filepath_full = os.path.join(OUTPUT_DIR, HTML_BASENAME)

    # 1. Load and Preprocess Data
    train_data, test_prompts = load_data(INPUT_DATA_FILE)
    train_texts, train_labels_encoded, test_texts, test_labels_encoded, label_encoder = preprocess_data(train_data, test_prompts)

    # 2. Run Benchmarks for each model
    all_results = []
    for model_name in MODEL_NAMES:
        if not isinstance(model_name, str) or '/' not in model_name:
             print(f"\nWARNING: Invalid model name format: '{model_name}'. Skipping.")
             continue
        result = run_benchmark(model_name, train_texts, train_labels_encoded, test_texts, test_labels_encoded, DEVICE)
        if result: all_results.append(result)
        else: print(f"Benchmark FAILED for model: {model_name}")
        print("-" * 70)

    # 3. Print Summary Table to Console
    print("\n\n" + "="*30 + " Benchmark Summary (Console) " + "="*30)
    if not all_results:
        print("No models completed the benchmark successfully.")
    else:
        all_results.sort(key=lambda x: (x.get('inference_tokens_per_second', 0), x.get('accuracy', 0)), reverse=True)
        header = f"{'Model Name':<45} | {'Size(MB)':>9} | {'Dim':>5} | {'Acc':>8} | {'Train Tok/s':>12} | {'Infer Tok/s':>12} | {'Inf ms/q':>10}"
        print(header); print("-" * len(header))
        for r in all_results:
            model_str = r.get('model_name', 'Unknown'); size_val = r.get('approx_size_mb', '?')
            size_str = f"{size_val:,}" if isinstance(size_val, (int, float)) else str(size_val)
            dim_str = str(r.get('embedding_dim', '?')); acc_str = f"{r.get('accuracy', 0):.4f}"
            train_tok_sec_val = r.get('train_tokens_per_second', 0)
            train_tok_sec_str = f"{train_tok_sec_val:,.0f}" if train_tok_sec_val > 0 else ("N/A" if train_tok_sec_val < 0 else "0")
            infer_tok_sec_val = r.get('inference_tokens_per_second', 0)
            infer_tok_sec_str = f"{infer_tok_sec_val:,.0f}" if infer_tok_sec_val > 0 else ("N/A" if infer_tok_sec_val < 0 else "0")
            inf_ms_str = f"{r.get('avg_inference_time_ms', 0):.1f}"
            print(f"{model_str:<45} | {size_str:>9} | {dim_str:>5} | {acc_str:>8} | {train_tok_sec_str:>12} | {infer_tok_sec_str:>12} | {inf_ms_str:>10}")

        # 4. Generate and Save Plot
        # plot_results now saves plot to output dir and returns path
        saved_plot_path = plot_results(all_results, OUTPUT_DIR, PLOT_BASENAME)

        # 5. Generate and Save HTML Report
        # Pass the path to the saved plot
        generate_html_report(all_results, saved_plot_path, html_filepath_full)


    # 6. Print Total Time
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    print(f"\nBenchmark finished in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")
    print(f"Results saved in directory: '{OUTPUT_DIR}'")
