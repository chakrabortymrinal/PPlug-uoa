
#!/bin/bash
set -e

echo "=== PPlug Resource Download Script ==="

# Decide where to store resources
BASE_DIR="$(pwd)"
MODEL_DIR="${BASE_DIR}"
DATA_DIR="${BASE_DIR}"

# Set SSL certs for both curl & Hugging Face CLI
export SSL_CERT_FILE="/Users/in22339881/Documents/custom-ca-bundle.pem"
export REQUESTS_CA_BUNDLE="/Users/in22339881/Documents/custom-ca-bundle.pem"

# ---------------- LaMP-3 Dataset ----------------
echo "[1/4] Checking LaMP-3 dataset..."
if [ -f "${DATA_DIR}/LaMP_time_3/train_questions.json" ] && \
   [ -f "${DATA_DIR}/LaMP_time_3/train_outputs.json" ] && \
   [ -f "${DATA_DIR}/LaMP_time_3/dev_questions.json" ] && \
   [ -f "${DATA_DIR}/LaMP_time_3/dev_outputs.json" ]; then
    echo "✅ LaMP-3 dataset already exists. Skipping download."
else
    echo "⬇ Downloading LaMP-3 dataset..."
    mkdir -p "${DATA_DIR}/LaMP_time_3"
    curl -L -o "${DATA_DIR}/LaMP_time_3/train_questions.json" https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_3/train/train_questions.json
    curl -L -o "${DATA_DIR}/LaMP_time_3/train_outputs.json" https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_3/train/train_outputs.json
    curl -L -o "${DATA_DIR}/LaMP_time_3/dev_questions.json" https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_3/dev/dev_questions.json
    curl -L -o "${DATA_DIR}/LaMP_time_3/dev_outputs.json" https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_3/dev/dev_outputs.json
fi

# ---------------- FlanT5-Large Model ----------------
echo "[2/3] Checking FlanT5-Large model..."
if [ -d "${MODEL_DIR}/FlanT5-Large" ] && [ "$(ls -A ${MODEL_DIR}/FlanT5-Large 2>/dev/null)" ]; then
    echo "✅ FlanT5-Large model already exists. Skipping download."
else
    echo "⬇ Downloading FlanT5-Large model..."
    mkdir -p "${MODEL_DIR}/FlanT5-Large"
    hf download google/flan-t5-large --local-dir "${MODEL_DIR}/FlanT5-Large"
fi

# ---------------- BGE Base Model ----------------
echo "[3/3] Checking bge-base-en-v1.5 model..."
if [ -d "${MODEL_DIR}/bge-base-en-v1.5" ] && [ "$(ls -A ${MODEL_DIR}/bge-base-en-v1.5 2>/dev/null)" ]; then
    echo "✅ bge-base-en-v1.5 model already exists. Skipping download."
else
    echo "⬇ Downloading bge-base-en-v1.5 model..."
    mkdir -p "${MODEL_DIR}/bge-base-en-v1.5"
    hf download BAAI/bge-base-en-v1.5 --local-dir "${MODEL_DIR}/bge-base-en-v1.5"
fi


# ---------------- BGE Base Model ----------------

echo "[3/3] Checking BAAI/bge-small-en-v1.5 model..."
MODEL_PATH="${MODEL_DIR}/bge-small-en-v1.5"

if [ -d "$MODEL_PATH" ] && [ "$(find "$MODEL_PATH" -mindepth 1 | head -n 1)" ]; then
    echo "✅ bge-small-en-v1.5 model already exists. Skipping download."
else
    echo "⬇ Downloading bge-small-en-v1.5 model..."
    mkdir -p "$MODEL_PATH"
    hf download BAAI/bge-small-en-v1.5 --local-dir "$MODEL_PATH" || { echo "❌ Download failed"; exit 1; }
fi

# ---------------- FlanT5-base Model ----------------
echo "[2/3] Checking FlanT5-base model..."
if [ -d "${MODEL_DIR}/FlanT5-base" ] && [ "$(ls -A ${MODEL_DIR}/FlanT5-base 2>/dev/null)" ]; then
    echo "✅ FlanT5-base model already exists. Skipping download."
else
    echo "⬇ Downloading FlanT5-base model..."
    mkdir -p "${MODEL_DIR}/FlanT5-base"
    hf download google/flan-t5-base --local-dir "${MODEL_DIR}/FlanT5-base"
fi


# ---------------- FlanT5-small Model ----------------
echo "[2/3] Checking FlanT5-small model..."
if [ -d "${MODEL_DIR}/FlanT5-small" ] && [ "$(ls -A ${MODEL_DIR}/FlanT5-small 2>/dev/null)" ]; then
    echo "✅ FlanT5-small model already exists. Skipping download."
else
    echo "⬇ Downloading FlanT5-small model..."
    mkdir -p "${MODEL_DIR}/FlanT5-small"
    hf download google/flan-t5-small --local-dir "${MODEL_DIR}/FlanT5-small"
fi


# ---------- DistilBERT base uncased ----------
echo "[4/4] Checking distilbert-base-uncased model..."
if [[ -d "${MODEL_DIR}/distilbert-base-uncased" ]] && [[ "$(ls -A "${MODEL_DIR}/distilbert-base-uncased" 2>/dev/null)" ]]; then
    echo "✅ distilbert-base-uncased model already exists. Skipping download."
else
    echo "⬇ Downloading distilbert-base-uncased model..."
    mkdir -p "${MODEL_DIR}/distilbert-base-uncased"
    hf download distilbert/distilbert-base-uncased --local-dir "${MODEL_DIR}/distilbert-base-uncased" || {
      echo "⚠  Failed to download distilbert-base-uncased — continuing"
    }
fi

echo "=== All checks complete ==="
