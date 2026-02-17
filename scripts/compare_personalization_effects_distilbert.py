"""
Compare outputs of:
  (A) DistilBERT baseline (no personalization)
  (B) Personalized DistilBERT using user profile embeddings
  (C) Personalized + Graph-enhanced DistilBERT (profile + GNN embeddings)

Purpose:
  Demonstrate how personalization and graph context modify rating classification.
  Monitor gate values to understand personalization strength.
"""

import os, sys, json, torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer, AutoModel, TrainingArguments
import torch.nn.functional as F

torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
# Setup imports
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extention.ModelForPer_distilbert_GNN_rating import PersonalBertRating

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
TASK_ID = 3
ENCODER_MODEL_PATH = "../distilbert-base-uncased"  # or your encoder path
CHECKPOINT_PATH = "../outputs_distilbert/task_3/checkpoint-1029"
MAX_HIS_LEN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------
# Training Arguments (Fixed Configuration)
# -------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=f"../outputs_distilbert/task_{TASK_ID}",
    
    # ✅ FIX: Learning rate schedule
    learning_rate=3e-5,  # ← Lower than before (was likely 5e-5)
    warmup_ratio=0.1,    # ← 10% warmup prevents early overfitting
    
    # ✅ FIX: Regularization
    weight_decay=0.01,   # ← Add weight decay for better generalization
    
    # ✅ FIX: Batch size and gradient accumulation
    per_device_train_batch_size=16,  # ← Smaller batch = more updates
    gradient_accumulation_steps=2,   # ← Effective batch = 32
    
    # ✅ FIX: Training length
    num_train_epochs=5,  # ← More epochs (you stopped at 2.99)
    
    # ✅ FIX: Evaluation strategy
    eval_strategy="steps",
    eval_steps=100,      # ← Evaluate more frequently
    save_steps=100,
    save_total_limit=3,  # ← Keep only best 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_acc",  # ← Optimize for accuracy
    greater_is_better=True,
    
    # ✅ FIX: Stability
    fp16=True if torch.cuda.is_available() else False,
    logging_steps=10,
    seed=42,
)

# Demo modes
SHOW_BASE = True
SHOW_PERSONA_REAL = True
SHOW_PERSONA_EMPTY = False
SHOW_GRAPH = True

# Dev evaluation
USE_DEV_GOLD = True
DEV_SUBSET = True
DEV_SAMPLE_SIZE = 10

# -------------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------------
encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_PATH)
encoder_model = AutoModel.from_pretrained(ENCODER_MODEL_PATH).to(DEVICE).eval()

# Baseline: encoder + simple classifier
baseline_classifier = torch.nn.Linear(encoder_model.config.hidden_size, 5).to(DEVICE).eval()

# Personalized model
personal_model = PersonalBertRating(
    encoder_model=encoder_model,
    task_id=TASK_ID,
    bge_emb_dir="../bge_emb",
    graph_emb_dir="../graph_emb",
    use_profile=True,
    use_session=True,
    use_graph=True,
).to(DEVICE).eval()

# Load checkpoint if exists
if os.path.exists(CHECKPOINT_PATH):
    # ✅ Try SafeTensors first, fall back to PyTorch .bin
    safetensors_path = os.path.join(CHECKPOINT_PATH, "model.safetensors")
    pytorch_bin_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        print(f"✅ Found SafeTensors checkpoint at {safetensors_path}")
        from safetensors.torch import load_file
        state = load_file(safetensors_path, device=str(DEVICE))
        ckpt_format = "safetensors"
    elif os.path.exists(pytorch_bin_path):
        print(f"✅ Found PyTorch checkpoint at {pytorch_bin_path}")
        state = torch.load(pytorch_bin_path, map_location=DEVICE)
        ckpt_format = "pytorch"
    else:
        print(f"❌ ERROR: No checkpoint found at {CHECKPOINT_PATH}")
        print(f"   Looked for: model.safetensors or pytorch_model.bin")
        sys.exit(1)
    
    # ✅ Check if classifier weights will change
    classifier_weight_before = personal_model.classifier.weight.data.clone()
    
    # ✅ Load with strict=False and inspect mismatches
    missing, unexpected = personal_model.load_state_dict(state, strict=False)
    
    classifier_weight_after = personal_model.classifier.weight.data
    weight_changed = not torch.allclose(classifier_weight_before, classifier_weight_after)
    
    print(f"✅ Loaded personalized model from {CHECKPOINT_PATH} ({ckpt_format})")
    print(f"   Missing keys: {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
    print(f"   Classifier weights changed: {weight_changed}")
    
    if not weight_changed:
        print("⚠️ WARNING: Classifier weights did NOT change after loading!")
        print("   This suggests the checkpoint may be corrupted or incompatible.")
    
    # ✅ Show key parameters loaded
    if missing:
        print(f"\n⚠️ Missing keys (model has these, checkpoint doesn't):")
        for key in list(missing)[:10]:  # Show first 10
            print(f"   - {key}")
    
    if unexpected:
        print(f"\n⚠️ Unexpected keys (checkpoint has these, model doesn't):")
        for key in list(unexpected)[:10]:
            print(f"   - {key}")
    
    # ✅ Critical parameters to verify
    critical_params = [
        "classifier.weight",
        "classifier.bias",
        "gate.weight",
        "gate.bias",
        "align_profile.0.weight"
    ]
    
    print("\n🔍 Critical parameters check:")
    for param_name in critical_params:
        if param_name in state:
            param_shape = state[param_name].shape
            print(f"   ✅ {param_name}: {param_shape}")
        else:
            print(f"   ❌ MISSING: {param_name}")
else:
    print(f"❌ ERROR: Checkpoint directory not found at {CHECKPOINT_PATH}")
    sys.exit(1)

# -------------------------------------------------------------------------
# Build test data (similar to T5 script)
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    if DEV_SUBSET:
        questions_path = "../LaMP_time_3_subset/dev_questions.json"
        outputs_path = "../LaMP_time_3_subset/dev_outputs.json"
    else:
        questions_path = "../LaMP_time_3/dev_questions.json"
        outputs_path = "../LaMP_time_3/dev_outputs.json"

    with open(questions_path, "r") as fq:
        questions_data = json.load(fq)
    with open(outputs_path, "r") as fo:
        outputs_data = json.load(fo)

    id_to_gold = {g["id"]: int(g["output"]) - 1 for g in outputs_data.get("golds", [])}  # 0-indexed
    candidates = [q for q in questions_data if q.get("id") in id_to_gold]
    sampled = random.sample(candidates, k=min(DEV_SAMPLE_SIZE, len(candidates)))

    test_items = []
    for q in sampled:
        qid = q.get("id")
        prompt = q.get("input", "")
        gold = id_to_gold.get(qid)
        test_items.append({"id": qid, "prompt": prompt, "gold": gold})
else:
    test_items = [
        {"id": None, "prompt": "The story was fun but predictable.", "gold": None},
        {"id": None, "prompt": "The shipping was fast!", "gold": None},
    ]

# -------------------------------------------------------------------------
# Build user history (same as T5 script)
# -------------------------------------------------------------------------
with open("../bge_emb/task_3_train_offsets.json") as f:
    offsets = json.load(f)

user_entry = next((e for e in offsets["entries"] if len(e["profile_id"]) >= MAX_HIS_LEN), None)
if user_entry is None:
    raise ValueError("No user found with enough profile reviews in offsets file.")

# ✅ FIX: Convert string IDs to integers
profile_his_ids_raw = user_entry["profile_id"][:MAX_HIS_LEN]
profile_his_ids = []
for hid in profile_his_ids_raw:
    try:
        profile_his_ids.append(int(hid))
    except (TypeError, ValueError):
        print(f"⚠️ Warning: Could not convert profile ID '{hid}' to int, using 0")
        profile_his_ids.append(0)

# Pad to MAX_HIS_LEN
if len(profile_his_ids) < MAX_HIS_LEN:
    profile_his_ids = profile_his_ids + [0] * (MAX_HIS_LEN - len(profile_his_ids))

his_id_real = torch.tensor([profile_his_ids], dtype=torch.long, device=DEVICE)
his_id_empty = torch.zeros_like(his_id_real)

print(f"✅ Using user with {len(profile_his_ids_raw)} profile reviews")
print(f"   Profile IDs (first 5): {profile_his_ids[:5]}")

# ✅ NEW: Verify IDs are in valid range
with open("../bge_emb/task_3_train_offsets.json") as f:
    offsets_data = json.load(f)
    
    # ✅ Determine max valid ID from offsets structure
    if "num_samples" in offsets_data:
        max_valid_id = offsets_data["num_samples"]
    elif "entries" in offsets_data:
        # Calculate from entries: max of all profile_ids
        all_ids = []
        for entry in offsets_data["entries"]:
            if "profile_id" in entry:
                for pid in entry["profile_id"]:
                    try:
                        all_ids.append(int(pid))
                    except (TypeError, ValueError):
                        pass
        max_valid_id = max(all_ids) + 1 if all_ids else 10000000  # fallback
        print(f"ℹ️ Calculated max_valid_id from entries: {max_valid_id}")
    else:
        # Fallback: use memmap size
        try:
            memmap_path = "../offline_cache_lamp3/task_3_his_train.mmap"
            if os.path.exists(memmap_path):
                import numpy as np
                memmap_data = np.memmap(memmap_path, dtype='float32', mode='r')
                # Assuming shape is (num_samples, embedding_dim)
                # Common embedding dims: 768, 1024
                for emb_dim in [768, 1024, 384]:
                    if len(memmap_data) % emb_dim == 0:
                        max_valid_id = len(memmap_data) // emb_dim
                        print(f"ℹ️ Calculated max_valid_id from memmap: {max_valid_id} (dim={emb_dim})")
                        break
                else:
                    max_valid_id = 10000000  # safe fallback
                    print(f"⚠️ Could not determine memmap shape, using fallback: {max_valid_id}")
            else:
                max_valid_id = 10000000
                print(f"⚠️ Memmap not found, using fallback: {max_valid_id}")
        except Exception as e:
            max_valid_id = 10000000
            print(f"⚠️ Error reading memmap: {e}, using fallback: {max_valid_id}")

invalid_ids = [hid for hid in profile_his_ids if hid < 0 or hid >= max_valid_id]
if invalid_ids:
    print(f"⚠️ WARNING: Found {len(invalid_ids)} invalid profile IDs (out of range)")
    print(f"   Valid range: [0, {max_valid_id})")
    print(f"   Invalid IDs: {invalid_ids[:10]}")
else:
    print(f"✅ All profile IDs are valid (range: [0, {max_valid_id}))")

# -------------------------------------------------------------------------
# Inference functions
# -------------------------------------------------------------------------
def run_baseline(prompt: str):
    """Run baseline DistilBERT classifier (no personalization)"""
    tokens = encoder_tokenizer(prompt, return_tensors="pt", max_length=256, 
                              truncation=True, padding=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = encoder_model(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = baseline_classifier(cls_emb)  # (1, 5)
        
    probs = F.softmax(logits, dim=-1)[0]  # (5,)
    pred_class = logits.argmax(dim=-1).item()
    
    return pred_class, probs.cpu().numpy(), logits[0].cpu().numpy()

def run_personalized(prompt: str, his_id: torch.Tensor, 
                     graph_node_ids=None, session_ids=None):
    """Run personalized DistilBERT model"""
    tokens = encoder_tokenizer(prompt, return_tensors="pt", max_length=256,
                              truncation=True, padding=True).to(DEVICE)
    
    with torch.no_grad():
        output = personal_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            his_id=his_id,
            graph_node_ids=graph_node_ids,
            session_ids=session_ids,
            force_dev_bank=True,  # ✅ ADD THIS
        )
    
    logits = output.logits[0]  # (5,)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    pred_class = logits.argmax(dim=-1).item()
    
    # ✅ ADD: Retrieve gate values
    gate_val = getattr(personal_model, '_last_gate_values', None)
    if gate_val is not None:
        gate_mean = float(gate_val.mean().item())
    else:
        gate_mean = None
    
    return pred_class, probs, logits.cpu().numpy(), gate_mean

# -------------------------------------------------------------------------
# Run comparison
# -------------------------------------------------------------------------

# ✅ ADD: Test both training and eval mode
print("\n🔍 Testing model in both modes:")

# Eval mode (current)
personal_model.eval()
pred_eval, probs_eval, _, gate_eval = run_personalized(test_items[0]["prompt"], his_id_real)

# Training mode (for comparison)
personal_model.train()
pred_train, probs_train, _, gate_train = run_personalized(test_items[0]["prompt"], his_id_real)

print(f"  EVAL mode: pred={pred_eval+1} | gate={gate_eval:.3f}")
print(f"  TRAIN mode: pred={pred_train+1} | gate={gate_train:.3f}")

# Reset to eval
personal_model.eval()

if pred_eval == pred_train and abs(gate_eval - gate_train) < 0.01:
    print("  ⚠️ WARNING: Model behaves identically in train/eval → possible gradient/dropout issue")

rows = []

for item in test_items:
    prompt = item["prompt"]
    gold = item["gold"]
    qid = item["id"]
    
    print(f"\n🧾 ID: {qid} | Prompt: {prompt[:80]}...")
    if gold is not None:
        print(f"  GOLD → {gold + 1}")  # Convert back to 1-5 for display
    
    results = {}
    
    if SHOW_BASE:
        pred, probs, logits = run_baseline(prompt)
        print(f"  BASE → {pred + 1} | probs: {probs} | confidence: {probs[pred]:.3f}")
        results["base"] = pred
        results["base_probs"] = probs
    
    if SHOW_PERSONA_REAL:
        pred, probs, logits, gate_val = run_personalized(prompt, his_id_real)
        gate_str = f"gate={gate_val:.3f}" if gate_val is not None else "gate=N/A"
        print(f"  PERS (REAL) → {pred + 1} | {gate_str} | probs: {probs} | conf: {probs[pred]:.3f}")
        results["pers"] = pred
        results["pers_probs"] = probs
        results["pers_gate"] = gate_val
    
    if SHOW_PERSONA_EMPTY:
        pred, probs, logits, gate_val = run_personalized(prompt, his_id_empty)
        print(f"  PERS (EMPTY) → {pred + 1} | probs: {probs} | confidence: {probs[pred]:.3f}")
        results["empty"] = pred
    
    if SHOW_GRAPH:
        # Build graph node IDs (similar to T5 script)
        # ... (add graph building logic here)
        pred, probs, logits, gate_val = run_personalized(prompt, his_id_real, 
                                               graph_node_ids=None)  # Add actual graph IDs
        print(f"  PERS + GRAPH → {pred + 1} | probs: {probs} | confidence: {probs[pred]:.3f}")
        results["graph"] = pred
    
    results.update({"id": qid, "gold": gold})
    rows.append(results)

# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    def compute_metrics(name, pred_key):
        golds = [r["gold"] for r in rows if r["gold"] is not None and pred_key in r]
        preds = [r[pred_key] for r in rows if r["gold"] is not None and pred_key in r]
        
        if not golds:
            print(f"\n=== METRICS: {name} ===")
            print("  (no valid predictions)")
            return
        
        # Convert 0-indexed to 1-5 for RMSE/MAE
        golds_1to5 = [g + 1 for g in golds]
        preds_1to5 = [p + 1 for p in preds]
        
        acc = accuracy_score(golds, preds)
        rmse = mean_squared_error(golds_1to5, preds_1to5, squared=False)
        mae = mean_absolute_error(golds_1to5, preds_1to5)
        
        print(f"\n=== METRICS: {name} ===")
        print(f"  n={len(golds)} | acc={acc:.3f} | rmse={rmse:.3f} | mae={mae:.3f}")
    
    if SHOW_BASE:
        compute_metrics("BASE", "base")
    if SHOW_PERSONA_REAL:
        compute_metrics("PERSONALIZED (REAL)", "pers")
    if SHOW_PERSONA_EMPTY:
        compute_metrics("PERSONALIZED (EMPTY)", "empty")
    if SHOW_GRAPH:
        compute_metrics("PERSONALIZED + GRAPH", "graph")