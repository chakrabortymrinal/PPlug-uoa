"""
Compare outputs of:
  (A) Flan‑T5 baseline (no personalization)
  (B) Stage A/B personalized model using profile history embeddings (+ optional graph)

This version is compatible with:
  extention/ModelForPer_slim_GNN_stageA_B.py (PersonalLLM_Slim_StageAB)

Notes:
  - StageAB currently exposes forward() returning {"loss","logits"} (no custom generate()).
    We therefore use a 1-step forward pass and read the predicted rating from logits[:,0,:].
  - This script intentionally focuses on numeric rating prediction (1..5) as a single token.
"""

import os
import sys
import json
import random
import re
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel

torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
# ✅ Setup imports
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from extention.ModelForPer_slim_GNN_stageA_B import PersonalLLM_Slim_StageAB

# -------------------------------------------------------------------------
# 🔧 CONFIG
# -------------------------------------------------------------------------
TASK_ID = 3
CHECKPOINT_PATH = "../extention/output_3/checkpoint-129"  # <-- change to your StageAB checkpoint folder
MAX_HIS_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ NEW: Add generation parameters to match training config
# These compensate for the new gate bias and temperature settings
GEN_MAX_NEW_TOKENS = 2
GEN_MIN_NEW_TOKENS = 1
GEN_NUM_BEAMS = 1  # Use 1 for deterministic single-token prediction
GEN_TEMPERATURE = 1.0  # Keep at 1.0 for rating tasks (no sampling)
GEN_DO_SAMPLE = False  # Disable sampling for rating prediction
GEN_TOP_P = 1.0  # Not used when do_sample=False

# ✅ NEW: Gate monitoring diagnostics
SHOW_GATE_DIAGNOSTICS = True
GATE_THRESHOLD_WARNING = 0.95  # Warn if gate is too open (personalization may override task)

# Evaluate on dev gold (recommended)
USE_DEV_GOLD = True
DEV_SUBSET = True
DEV_SAMPLE_SIZE = 200

# Demo controls
SHOW_BASE = True
SHOW_PERSONA_REAL = True
SHOW_PERSONA_EMPTY = True
SHOW_GRAPH = True

# Optional: restrict users for qualitative inspection (leave empty set() to disable)
INCLUDE_USER_IDS = set()

# -------------------------------------------------------------------------
# 🧠 LOAD MODELS
# -------------------------------------------------------------------------
llm_model_path = "../FlanT5-base"
emb_model_path = "../bge-base-en-v1.5"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path, use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_path)

baseline_llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_path).to(DEVICE).eval()
pers_llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_path).to(DEVICE).eval()
emb_model = AutoModel.from_pretrained(emb_model_path).to(DEVICE).eval()

# -------------------------------------------------------------------------
# 🚀 LOAD StageAB MODEL + CHECKPOINT
# -------------------------------------------------------------------------
personal_model = PersonalLLM_Slim_StageAB(
    llm_model=pers_llm_model,
    emb_model=emb_model,
    llm_tokenizer=llm_tokenizer,
    max_input_len=256,
    max_new_len=64,
    task_id=TASK_ID,
    use_profile=True,
    use_session=False,  # turn on only if you also supply session_ids
    use_graph=True,
).to(DEVICE).eval()

# ✅ FIX: Resize LLM token embeddings to match checkpoint vocab BEFORE loading
checkpoint_vocab_size = 32100  # from the error / your trained checkpoint
current_vocab_size = personal_model.llm_model.shared.weight.shape[0]
if current_vocab_size != checkpoint_vocab_size:
    print(f"⚠️  Vocab size mismatch: current={current_vocab_size}, checkpoint={checkpoint_vocab_size}")
    print(f"   Resizing model embeddings to {checkpoint_vocab_size}...")
    personal_model.llm_model.resize_token_embeddings(checkpoint_vocab_size)
    print("✅ Model resized to match checkpoint")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"❌ Checkpoint directory not found: {CHECKPOINT_PATH}")

print(f"📦 Loading StageAB checkpoint from {CHECKPOINT_PATH}")
loaded = False
for ckpt_file in ["pytorch_model.bin", "model.pt", "checkpoint.pt"]:
    ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_file)
    if not os.path.exists(ckpt_path):
        continue

    print(f"  Found: {ckpt_file}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    # Handle wrapped formats
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]

    missing, unexpected = personal_model.load_state_dict(checkpoint, strict=False)
    print(f"  ✅ Weights loaded (strict=False)")
    if missing:
        print(f"  ⚠️  Missing keys: {len(missing)} (showing first 10)")
        for k in missing[:10]:
            print(f"    - {k}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {len(unexpected)} (showing first 10)")
        for k in unexpected[:10]:
            print(f"    - {k}")

    loaded = True
    break

if not loaded:
    raise FileNotFoundError(f"No checkpoint file found in {CHECKPOINT_PATH}")

print("✅ StageAB model ready for inference\n")

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def extract_numeric_rating(text: str) -> Optional[int]:
    m = re.search(r"\b([1-5])(\.0+)?\b", str(text))
    return int(m.group(1)) if m else None

def _rating_token_ids() -> Dict[str, int]:
    ids: Dict[str, int] = {}
    for d in ["1", "2", "3", "4", "5"]:
        enc = llm_tokenizer.encode(d, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"Digit '{d}' is not a single token for this tokenizer: {enc}")
        ids[d] = enc[0]
    return ids

_DIGIT_TOKEN_IDS = _rating_token_ids()

def rating_probs_from_first_step(logits_1step: torch.Tensor) -> Dict[str, Any]:
    if logits_1step.dim() == 2:
        logits_1step = logits_1step[0]  # (V,)

    digit_ids = torch.tensor([_DIGIT_TOKEN_IDS[str(i)] for i in range(1, 6)],
                             device=logits_1step.device, dtype=torch.long)
    digit_logits = logits_1step.index_select(dim=0, index=digit_ids)  # (5,)
    probs = F.softmax(digit_logits, dim=-1)  # (5,)

    vals = probs.detach().cpu().tolist()
    items = {str(i + 1): float(vals[i]) for i in range(5)}
    top2 = sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:2]
    top_choice, top_prob = top2[0]
    second_prob = top2[1][1] if len(top2) > 1 else 0.0

    entropy = float(-(probs * (probs + 1e-12).log()).sum().detach().cpu().item())
    margin = float(top_prob - second_prob)
    return {"p": items, "top": top_choice, "top_prob": float(top_prob), "margin": margin, "entropy": entropy}

def run_llm_base_with_transparency(prompt: str) -> Tuple[str, str, Dict[str, Any]]:
    tokens = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    gen = baseline_llm_model.generate(
        **tokens,
        num_beams=3,
        max_new_tokens=2,
        min_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
    first_step_logits = gen.scores[0]  # (1, V)
    info = rating_probs_from_first_step(first_step_logits)
    text = llm_tokenizer.decode(gen.sequences[0], skip_special_tokens=True).strip()
    return info["top"], text, info

def run_stageab_forward_1step(
    prompt: str,
    his_id: torch.Tensor,
    graph_node_ids: Optional[torch.Tensor],
    return_diagnostics: bool = True
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Run StageAB forward() and extract rating prediction + diagnostics.
    
    ✅ UPDATED: Now monitors gate values and checks for persona override
    """
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    dummy_labels = torch.full_like(llm_inp["input_ids"], fill_value=-100)

    graph_node_mask = None
    if graph_node_ids is not None:
        graph_node_mask = graph_node_ids.ge(0).long()

    # ✅ Clear previous gate diagnostics before forward pass
    personal_model._gate_means = []

    with torch.no_grad():
        out = personal_model(
            llm_input_ids=llm_inp["input_ids"],
            llm_attention_mask=llm_inp["attention_mask"],
            labels=dummy_labels,
            emb_input_ids=emb_inp["input_ids"],
            emb_attention_mask=emb_inp["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
            his_id=his_id,
            session_ids=None,
            graph_node_ids=graph_node_ids,
            graph_node_mask=graph_node_mask,
        )

    logits = out.get("logits", None)
    if logits is None or (not torch.is_tensor(logits)) or logits.dim() != 3:
        return "(no_logits)", "", None

    # Extract rating prediction
    logits_1step = logits[:, 0, :]  # (B, V)
    info = rating_probs_from_first_step(logits_1step[0])
    rating = info["top"]

    # ✅ NEW: Add gate diagnostics
    if return_diagnostics and hasattr(personal_model, '_gate_means') and personal_model._gate_means:
        gate_mean = float(np.mean(personal_model._gate_means))
        info["gate_mean"] = gate_mean
        
        # ✅ Warn if gate is too open (persona may override task understanding)
        if gate_mean > GATE_THRESHOLD_WARNING:
            info["gate_warning"] = f"Gate very open ({gate_mean:.3f}) - personalization may override task"
        elif gate_mean < 0.05:
            info["gate_warning"] = f"Gate nearly closed ({gate_mean:.3f}) - minimal personalization"
    
    # ✅ NEW: Check for probability distribution sanity
    top_prob = info.get("top_prob", 0.0)
    margin = info.get("margin", 0.0)
    
    if top_prob < 0.3:
        info["pred_warning"] = f"Low confidence prediction (top_prob={top_prob:.3f})"
    elif margin < 0.1:
        info["pred_warning"] = f"Ambiguous prediction (margin={margin:.3f})"

    return rating, rating, info

def _max_prob_delta(probs1: Dict[str, float], probs2: Dict[str, float]) -> float:
    """Helper to compute max absolute probability difference between two distributions"""
    keys = set(probs1.keys()) | set(probs2.keys())
    return max(abs(probs1.get(k, 0.0) - probs2.get(k, 0.0)) for k in keys)

# ✅ NEW: Enhanced diagnostics printing
def print_prediction_with_diagnostics(
    label: str,
    rating: str,
    text: str,
    info: Optional[Dict[str, Any]]
):
    """Print prediction with color-coded diagnostics"""
    print(f"  {label:30s} → {text}")
    
    if info is None:
        return
    
    # Rating distribution
    probs_str = " ".join([f"{k}:{info['p'][k]:.2f}" for k in sorted(info['p'].keys())])
    print(f"  [Probs: {probs_str}]")
    
    # Core metrics
    print(f"  [Margin={info['margin']:.3f} | Entropy={info['entropy']:.3f} | TopProb={info['top_prob']:.3f}]")
    
    # ✅ Gate diagnostics
    if 'gate_mean' in info:
        gate_status = "🟢" if 0.3 <= info['gate_mean'] <= 0.7 else "🟡" if 0.1 <= info['gate_mean'] < 0.3 or 0.7 < info['gate_mean'] <= 0.9 else "🔴"
        print(f"  {gate_status} [Gate={info['gate_mean']:.4f}]")
    
    # Warnings
    for warn_key in ['gate_warning', 'pred_warning']:
        if warn_key in info:
            print(f"  ⚠️  {info[warn_key]}")

# ✅ NEW: Enhanced metrics with statistical significance
def _metric_block_enhanced(name: str, get_pred, rows: List[Dict[str, Any]]):
    """
    Compute metrics with confidence analysis
    """
    gold_nums: List[int] = []
    pred_nums: List[int] = []
    confidences: List[float] = []
    gate_vals: List[float] = []
    
    for r in rows:
        if r["gold"] is None:
            continue
        g = extract_numeric_rating(r["gold"])
        p = extract_numeric_rating(get_pred(r))
        
        if g is not None and p is not None:
            gold_nums.append(g)
            pred_nums.append(p)
            
            # Extract confidence metrics
            info_key = name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "") + "_info"
            if info_key in r and r[info_key]:
                confidences.append(r[info_key].get('top_prob', 0.0))
                gate_vals.append(r[info_key].get('gate_mean', 0.0))

    print(f"\n{'='*60}")
    print(f"📊 METRICS: {name}")
    print(f"{'='*60}")
    
    if not gold_nums:
        print("  ❌ No parsable numeric pairs")
        return

    acc = float(np.mean([int(g == p) for g, p in zip(gold_nums, pred_nums)]))
    rmse = float(mean_squared_error(gold_nums, pred_nums, squared=False))
    mae = float(mean_absolute_error(gold_nums, pred_nums))
    
    print(f"  n={len(gold_nums)}")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  RMSE:      {rmse:.3f}")
    print(f"  MAE:       {mae:.3f}")
    
    # ✅ NEW: Confidence and gate analysis
    if confidences:
        avg_conf = float(np.mean(confidences))
        print(f"  Avg Confidence: {avg_conf:.3f}")
        
        if avg_conf < 0.4:
            print(f"    ⚠️  Low average confidence - model is uncertain")
    
    if gate_vals:
        avg_gate = float(np.mean(gate_vals))
        print(f"  Avg Gate:  {avg_gate:.3f}")
        
        if avg_gate > 0.9:
            print(f"    ⚠️  Gate very open - strong persona override (may ignore task context)")
        elif avg_gate < 0.1:
            print(f"    ⚠️  Gate nearly closed - minimal personalization effect")

# -------------------------------------------------------------------------
# Pick a user + history ids (same approach as your existing script)
# -------------------------------------------------------------------------
offsets_path = f"../bge_emb/task_{TASK_ID}_train_offsets.json"
if not os.path.exists(offsets_path):
    raise FileNotFoundError(f"Missing offsets file: {offsets_path}")

with open(offsets_path, "r") as f:
    offsets = json.load(f)

candidates = [e for e in offsets.get("entries", []) if len(e.get("profile_id", [])) >= MAX_HIS_LEN]
if INCLUDE_USER_IDS:
    filtered = []
    for e in candidates:
        pids = e.get("profile_id", [])
        if pids and str(pids[0]) in INCLUDE_USER_IDS:
            filtered.append(e)
    candidates = filtered or candidates

if not candidates:
    raise ValueError(f"No offsets entry has at least MAX_HIS_LEN={MAX_HIS_LEN} profiles.")

user_entry = random.choice(candidates)
start_idx = int(user_entry.get("start", 0))

his_row_ids = list(range(start_idx, start_idx + MAX_HIS_LEN))
profile_his_ids_raw = user_entry.get("profile_id", [])[:MAX_HIS_LEN]
# ✅ FIXED: Ensure profile_his_ids are strings for consistent lookup
profile_his_ids = [str(x) for x in profile_his_ids_raw]
his_id_real = torch.tensor([his_row_ids], dtype=torch.long, device=DEVICE)
his_id_empty = torch.zeros_like(his_id_real)

print(f"✅ Using offsets start_idx={start_idx} -> his_row_ids={his_row_ids}")
print(f"   Profile history IDs: {profile_his_ids}")

# ✅ NEW: Verify data structure
print(f"\n🔍 DATASET DIAGNOSTICS:")
print(f"   Offsets file: {offsets_path}")
print(f"   Total candidate entries: {len(candidates)}")
print(f"   Selected entry start_idx: {start_idx}")
print(f"   Number of profile_his_ids: {len(profile_his_ids)}")
print(f"   First 5 profile_his_ids: {profile_his_ids[:5]}")
print(f"   Data type of IDs: {type(profile_his_ids[0]) if profile_his_ids else 'empty'}")

# ============================================================================
# GRAPH NODE MAPPING - FIXED VERSION
# ============================================================================

his_to_graph_path = f"../graph_emb/task_{TASK_ID}_his_to_graph.json"
review_node_indices = []

if os.path.exists(his_to_graph_path):
    with open(his_to_graph_path, "r") as f:
        node_maps = json.load(f)
    
    print(f"\n📖 Graph mapping loaded: {len(node_maps)} entries")
    print(f"   Type: {type(node_maps)}")
    
    # ✅ FIXED: Properly extract graph node indices with string normalization
    valid_mappings = 0
    for i, hid in enumerate(profile_his_ids):
        # hid is already a string due to normalization above
        node_idx = -1
        
        # Strategy 1: Direct string key
        if hid in node_maps:
            node_idx = int(node_maps[hid])
            valid_mappings += 1
        # Strategy 2: Prefixed key (e.g., "review_9119734")
        elif f"review_{hid}" in node_maps:
            node_idx = int(node_maps[f"review_{hid}"])
            valid_mappings += 1
        
        review_node_indices.append(node_idx)
        
        # Debug first few mappings
        if i < 5:
            print(f"   profile_his_id[{i}] = '{hid}' → graph_node = {node_idx}")
    
    print(f"\n✅ Mapped {valid_mappings}/{len(profile_his_ids)} profile IDs to graph nodes")
    
    if valid_mappings == 0:
        print("❌ WARNING: NO VALID MAPPINGS FOUND!")
        print("   Checking key format mismatch...")
        print(f"   Sample node_maps keys: {list(node_maps.keys())[:10]}")
        print(f"   Sample profile_his_ids: {profile_his_ids[:10]}")
else:
    print(f"⚠️ Graph mapping file not found: {his_to_graph_path}")
    print("   → Graph pathway will be disabled")
    review_node_indices = [-1] * MAX_HIS_LEN

print(f"\n🔗 Review node indices (first 10): {review_node_indices[:10]}")
print(f"   Valid nodes: {sum(1 for x in review_node_indices if x >= 0)}/{len(review_node_indices)}")

graph_node_ids = torch.tensor([review_node_indices], dtype=torch.long, device=DEVICE)

# -------------------------------------------------------------------------
# Load dev gold prompts (or fallback)
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    if DEV_SUBSET:
        questions_path = f"../LaMP_time_{TASK_ID}_subset/dev_questions.json"
        outputs_path = f"../LaMP_time_{TASK_ID}_subset/dev_outputs.json"
    else:
        questions_path = f"../LaMP_time_{TASK_ID}/dev_questions.json"
        outputs_path = f"../LaMP_time_{TASK_ID}/dev_outputs.json"

    if not (os.path.exists(questions_path) and os.path.exists(outputs_path)):
        raise FileNotFoundError(f"Missing dev files: {questions_path} or {outputs_path}")

    with open(questions_path, "r") as fq:
        questions_data = json.load(fq)
    with open(outputs_path, "r") as fo:
        outputs_data = json.load(fo)

    # ✅ NEW: Verify loaded test data structure
    print(f"\n🔍 TEST DATA DIAGNOSTICS:")
    print(f"   Questions file: {questions_path}")
    print(f"   Outputs file: {outputs_path}")
    print(f"   Total test examples: {len(questions_data)}")
    
    if len(questions_data) > 0:
        first_example = questions_data[0]
        print(f"   First example keys: {first_example.keys()}")
        print(f"   First example ID: {first_example.get('id', 'N/A')}")
        
        if 'profile' in first_example:
            print(f"   First example profile: {str(first_example.get('profile', 'N/A'))[:100]}...")
            try:
                # Parse profile (could be string or already parsed)
                profile_raw = first_example['profile']
                if isinstance(profile_raw, str):
                    profile_data = json.loads(profile_raw)
                else:
                    profile_data = profile_raw
                
                # ✅ FIXED: Handle both dict and list formats
                if isinstance(profile_data, dict):
                    print(f"   Profile structure: dict with keys {list(profile_data.keys())}")
                    
                    if 'profile_his_ids' in profile_data:
                        his_ids = profile_data['profile_his_ids']
                        print(f"   Number of profile_his_ids: {len(his_ids)}")
                        print(f"   First 5 profile_his_ids: {his_ids[:5]}")
                        print(f"   Data type of IDs: {type(his_ids[0]) if his_ids else 'empty'}")
                    else:
                        print(f"   ⚠️  No 'profile_his_ids' key found")
                        
                elif isinstance(profile_data, list):
                    print(f"   Profile structure: list with {len(profile_data)} items")
                    
                    if len(profile_data) > 0:
                        # Check if list items are review IDs or structured objects
                        first_item = profile_data[0]
                        print(f"   First item type: {type(first_item)}")
                        print(f"   First item: {first_item}")
                        
                        if isinstance(first_item, dict):
                            print(f"   List contains dicts with keys: {list(first_item.keys())}")
                        else:
                            print(f"   List contains raw values (likely review IDs)")
                else:
                    print(f"   ⚠️  Unexpected profile structure: {type(profile_data)}")
                    
            except (json.JSONDecodeError, TypeError) as e:
                print(f"   ⚠️  Could not parse profile field: {e}")
        else:
            print(f"   No profile field in example")
    id_to_gold = {g["id"]: str(g["output"]).strip() for g in outputs_data.get("golds", [])}
    candidates_q = [q for q in questions_data if q.get("id") in id_to_gold]
    sampled = random.sample(candidates_q, k=min(DEV_SAMPLE_SIZE, len(candidates_q)))

    test_items = [{"id": q.get("id"), "prompt": q.get("input", ""), "gold": id_to_gold.get(q.get("id"), "")} for q in sampled]
else:
    test_items = [{
        "id": None,
        "prompt": "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The story was fun but predictable.",
        "gold": None
    }]

# -------------------------------------------------------------------------
# Run comparison
# -------------------------------------------------------------------------
rows: List[Dict[str, Any]] = []

for item in test_items:
    prompt = item["prompt"]
    gold = item["gold"]
    qid = item["id"]

    print(f"\n{'='*80}")
    print(f"🧾 ID: {qid}")
    print(f"📝 Prompt: {prompt[:100]}...")
    if gold is not None:
        print(f"🎯 GOLD: {gold}")
    print(f"{'='*80}")

    base_info = pers_info = emp_info = graph_info = None
    out_base = out_pers = out_emp = out_graph = None

    if SHOW_BASE:
        base_rating, base_text, base_info = run_llm_base_with_transparency(prompt)
        print_prediction_with_diagnostics("BASE (Flan-T5)", base_rating, base_text, base_info)
        out_base = base_rating

    if SHOW_PERSONA_REAL:
        pers_rating, pers_text, pers_info = run_stageab_forward_1step(
            prompt, his_id_real, graph_node_ids=None
        )
        print_prediction_with_diagnostics("StageAB (REAL)", pers_rating, pers_text, pers_info)
        out_pers = pers_rating

    if SHOW_PERSONA_EMPTY:
        emp_rating, emp_text, emp_info = run_stageab_forward_1step(
            prompt, his_id_empty, graph_node_ids=None
        )
        print_prediction_with_diagnostics("StageAB (EMPTY)", emp_rating, emp_text, emp_info)
        out_emp = emp_rating

    if SHOW_GRAPH:
        graph_node_ids_tensor = torch.tensor([review_node_indices], dtype=torch.long, device=DEVICE)
        
        # ✅ NEW: Validate graph nodes before inference
        valid_count = (graph_node_ids_tensor >= 0).sum().item()
        
        if valid_count == 0:
            print(f"  ⚠️  SKIPPING GRAPH PATH: No valid graph nodes (all -1)")
            print(f"     → REAL+GRAPH will be identical to REAL")
            graph_rating, graph_text, graph_info = pers_rating, pers_text, pers_info
        else:
            print(f"  🔗 Graph nodes: {valid_count}/{len(review_node_indices)} valid")
            
            graph_rating, graph_text, graph_info = run_stageab_forward_1step(
                prompt, his_id_real, graph_node_ids=graph_node_ids_tensor
            )
            
            print(f"  Personalized + Graph         → {graph_text}")
            if graph_info is not None:
                print(f"  [GRAPH rating/probs]         → rating={graph_rating} p={graph_info['p']} margin={graph_info['margin']:.3f} H={graph_info['entropy']:.3f}")
                
                # ✅ NEW: Compare graph vs non-graph outputs
                if pers_info is not None:
                    prob_delta = _max_prob_delta(graph_info['p'], pers_info['p'])
                    print(f"  📊 Δ(GRAPH vs REAL) prob:    {prob_delta:.6f}")
                    
                    if prob_delta < 0.01:
                        print(f"     ⚠️  Graph output nearly identical to REAL!")
                        print(f"     → Check if graph encoder is frozen or graph embeddings are zeros")
        
        out_graph = graph_rating

    if SHOW_GATE_DIAGNOSTICS:
        print(f"\n  📊 Cross-Model Comparison:")
        
        if pers_info and emp_info:
            prob_delta = sum(abs(pers_info['p'][k] - emp_info['p'][k]) for k in pers_info['p'])
            print(f"    Δ(REAL vs EMPTY) prob distribution: {prob_delta:.4f}")
            
            if prob_delta < 0.05:
                print(f"    ⚠️  WARNING: REAL and EMPTY produce nearly identical outputs!")
                print(f"    → Persona embeddings may not be properly loaded or utilized")
        
        if pers_info and base_info:
            pred_match = (pers_rating == base_rating)
            print(f"    REAL vs BASE predictions: {'🟰 MATCH' if pred_match else '≠ DIFFER'}")
            
            if pred_match and pers_info.get('gate_mean', 0) > 0.5:
                print(f"    ⚠️  Gate open but output matches base (personalization ineffective?)")

    rows.append({
        "id": qid,
        "gold": gold,
        "base": out_base,
        "pers": out_pers,
        "empty": out_emp,
        "graph": out_graph,
        "pers_info": pers_info,
        "emp_info": emp_info,
        "graph_info": graph_info,
    })

# -------------------------------------------------------------------------
# ✅ NEW: Analyze gate behavior across all samples
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("📊 GATE ACTIVATION ANALYSIS")
print("="*80)

all_gate_values = []
for row in rows:
    if row.get("pers_info") and "gate_mean" in row["pers_info"]:
        all_gate_values.append(row["pers_info"]["gate_mean"])

if all_gate_values:
    gate_mean = np.mean(all_gate_values)
    gate_std = np.std(all_gate_values)
    gate_min = np.min(all_gate_values)
    gate_max = np.max(all_gate_values)
    
    print(f"Gate statistics across {len(all_gate_values)} samples:")
    print(f"  Mean:  {gate_mean:.4f}")
    print(f"  Std:   {gate_std:.4f}")
    print(f"  Range: [{gate_min:.4f}, {gate_max:.4f}]")
    
    if gate_std < 0.01:
        print("\n⚠️  CRITICAL: Gate has near-zero variance!")
        print("   → Gate is NOT adapting to different inputs")
        print("   → Personalization strength is constant (ineffective)")
        print("\n📋 Possible causes:")
        print("   1. Gate was initialized with bias ~0.296 (sigmoid ≈ 0.57)")
        print("   2. Gate weights didn't train (frozen or learning rate too low)")
        print("   3. Gate inputs are too similar (profile embeddings collapse)")
    elif gate_std < 0.05:
        print("\n⚠️  Gate has low variance (adaptive personalization is weak)")
    else:
        print("\n✅ Gate is adapting across samples (healthy variance)")

# -------------------------------------------------------------------------
# Metrics (numeric match + RMSE/MAE)
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    if SHOW_BASE:
        _metric_block_enhanced("BASE", lambda r: r["base"], rows)
    if SHOW_PERSONA_REAL:
        _metric_block_enhanced("StageAB (REAL)", lambda r: r["pers"], rows)
    if SHOW_PERSONA_EMPTY:
        _metric_block_enhanced("StageAB (EMPTY)", lambda r: r["empty"], rows)
    if SHOW_GRAPH:
        _metric_block_enhanced("StageAB (REAL + GRAPH)", lambda r: r["graph"], rows)