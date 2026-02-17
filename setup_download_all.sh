
#!/bin/bash
set -euo pipefail

echo "=== PPlug Resource Download Script ==="

# Decide where to store resources
BASE_DIR="$(pwd)"
MODEL_DIR="${BASE_DIR}"
DATA_DIR="${BASE_DIR}"

# ----- Optional corporate CA bundle (uncomment if you need it) -----
# export SSL_CERT_FILE="/path/to/custom-ca-bundle.pem"
# export REQUESTS_CA_BUNDLE="/path/to/custom-ca-bundle.pem"

# Curl defaults: fail on HTTP errors, follow redirects, show progress
CURL_OPTS=( -fL --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 600 )

# ---------- Helper: download a single file if missing ----------
download_if_missing() {
  local url="$1"
  local dest="$2"

  # If file exists and is non-empty, skip
  if [[ -s "$dest" ]]; then
    echo "   ✅ Exists: $(basename "$dest") — skipping"
    return 0
  fi

  # Ensure destination directory exists
  mkdir -p "$(dirname "$dest")"

  echo "   ⬇ Downloading: $(basename "$dest")"
  if ! curl "${CURL_OPTS[@]}" -o "$dest.tmp" "$url"; then
    echo "   ⚠  Failed: $(basename "$dest") from $url — skipping"
    rm -f "$dest.tmp"
    return 1
  fi

  mv "$dest.tmp" "$dest"
  echo "   ✅ Saved: $(basename "$dest")"
}

# ---------- LaMP Time datasets (tasks 1–7) ----------
echo "[1/3] Checking LaMP Time datasets (1–7)..."
for task in {1..7}; do
  task_dir="${DATA_DIR}/LaMP_time_${task}"
  echo " - Task LaMP_${task} → ${task_dir}"

  base_url_train="https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_${task}/train"
  base_url_dev="https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_${task}/dev"

  # Expected files
  files=(
    "${base_url_train}/train_questions.json|${task_dir}/train_questions.json"
    "${base_url_train}/train_outputs.json|${task_dir}/train_outputs.json"
    "${base_url_dev}/dev_questions.json|${task_dir}/dev_questions.json"
    "${base_url_dev}/dev_outputs.json|${task_dir}/dev_outputs.json"
  )

  # Try each file; if a particular file isn’t available, we log and continue
  for spec in "${files[@]}"; do
    IFS="|" read -r url dest <<< "$spec"
    download_if_missing "$url" "$dest" || true
  done
done

# ---------- FlanT5-Large model ----------
echo "[2/3] Checking FlanT5-Large model..."
if [[ -d "${MODEL_DIR}/FlanT5-Large" ]] && [[ "$(ls -A "${MODEL_DIR}/FlanT5-Large" 2>/dev/null)" ]]; then
    echo "✅ FlanT5-Large model already exists. Skipping download."
else
    echo "⬇ Downloading FlanT5-Large model..."
    mkdir -p "${MODEL_DIR}/FlanT5-Large"
    # Requires Hugging Face CLI (`pip install huggingface_hub`), already implied in your setup
    hf download google/flan-t5-large --local-dir "${MODEL_DIR}/FlanT5-Large" || {
      echo "⚠  Failed to download FlanT5-Large — continuing"
    }
fi

# ---------- BGE base model ----------
echo "[3/3] Checking bge-base-en-v1.5 model..."
if [[ -d "${MODEL_DIR}/bge-base-en-v1.5" ]] && [[ "$(ls -A "${MODEL_DIR}/bge-base-en-v1.5" 2>/dev/null)" ]]; then
    echo "✅ bge-base-en-v1.5 model already exists. Skipping download."
else
    echo "⬇ Downloading bge-base-en-v1.5 model..."
    mkdir -p "${MODEL_DIR}/bge-base-en-v1.5"
    hf download BAAI/bge-base-en-v1.5 --local-dir "${MODEL_DIR}/bge-base-en-v1.5" || {
      echo "⚠  Failed to download bge-base-en-v1.5 — continuing"
    }
fi

echo "=== All checks complete ==="
