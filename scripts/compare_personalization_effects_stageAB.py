
"""
Compare outputs of:
  (A) Flan‑T5 baseline (no personalization)
  (B) Stage A/B personalized model using profile history embeddings (+ optional graph)

✅ UPDATED: Fixed import to use correct class name
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
# Setup imports
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ FIX: Use the correct class name from the module
from extention.ModelForPer_slim_GNN_stageA_B import (
    PersonalLLM_Slim_StageAB as PersonalizedModel,
    TrainHyperparams
)

print("✅ Successfully imported PersonalLLM_Slim_StageAB")

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
TASK_ID = "3"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
BASE_MODEL_NAME = "../flant5-large"
BGE_MODEL_NAME = "../bge-base-en-v1.5"
CHECKPOINT_PATH = "../extention/output_3/checkpoint-344"

# Data configuration
MAX_HIS_LEN = 3
USE_DEV_GOLD = True
DEV_SAMPLE_SIZE = 50
DEV_SUBSET = False
INCLUDE_USER_IDS = set()

# What to compare
SHOW_BASE = True
SHOW_PERSONA_REAL = True
SHOW_PERSONA_EMPTY = True
SHOW_GRAPH = True

# Generation parameters
MAX_NEW_TOKENS = 10
NUM_BEAMS = 4
TEMPERATURE = 1.0
TOP_P = 0.95

# Gate diagnostic thresholds
GATE_VERY_OPEN_THRESHOLD = 0.85
GATE_VERY_CLOSED_THRESHOLD = 0.15
GATE_BALANCED_RANGE = (0.3, 0.7)
PROB_DELTA_THRESHOLD = 0.01

# Checkpoint validation flags
VERIFY_MODEL_STATE = True

print(f"{'='*80}")
print(f"🔍 COMPARISON SCRIPT INITIALIZED")
print(f"{'='*80}")
print(f"Task ID: {TASK_ID}")
print(f"Device: {DEVICE}")
print(f"Base Model: {BASE_MODEL_NAME}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Max History Length: {MAX_HIS_LEN}")
print(f"Dev Sample Size: {DEV_SAMPLE_SIZE}")
print(f"Generation: beams={NUM_BEAMS}, temp={TEMPERATURE}, top_p={TOP_P}")
print(f"{'='*80}\n")

# -------------------------------------------------------------------------
# Digit token IDs for T5 tokenizer
# -------------------------------------------------------------------------
_temp_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
_DIGIT_TOKEN_IDS = {str(i): _temp_tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)}
del _temp_tokenizer

print(f"✅ Digit token IDs: {_DIGIT_TOKEN_IDS}")

# -------------------------------------------------------------------------
# Checkpoint verification
# -------------------------------------------------------------------------
def verify_checkpoint_state(checkpoint_path: str) -> Optional[Dict]:
    """Verify checkpoint contains expected components."""
    
    print(f"\n{'='*80}")
    print(f"🔍 CHECKPOINT VERIFICATION")
    print(f"{'='*80}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    files = os.listdir(checkpoint_path)
    print(f"\n📁 Checkpoint contents:")
    for f in sorted(files):
        file_path = os.path.join(checkpoint_path, f)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    
    required_files = ["model.safetensors", "config.json", "generation_config.json"]
    missing = [f for f in required_files if f not in files]
    
    if missing:
        print(f"\n⚠️  Missing files: {missing}")
    else:
        print(f"\n✅ All required files present")
    
    # Load training state if available
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        
        print(f"\n📊 Training metrics:")
        if 'log_history' in trainer_state and trainer_state['log_history']:
            last_log = trainer_state['log_history'][-1]
            
            for key in ['eval_accuracy', 'eval_loss', 'eval_rmse', 'eval_mae']:
                if key in last_log:
                    print(f"  {key}: {last_log[key]:.4f}")
        
        print(f"{'='*80}\n")
        return trainer_state
    
    print(f"{'='*80}\n")
    return None

# -------------------------------------------------------------------------
# Load hyperparameters
# -------------------------------------------------------------------------
def load_hyperparams_from_checkpoint(checkpoint_path: str) -> TrainHyperparams:
    """Load hyperparameters from checkpoint or use defaults."""
    hparams_path = os.path.join(checkpoint_path, "hyperparameters.json")
    
    if os.path.exists(hparams_path):
        print(f"📖 Loading hyperparameters from {hparams_path}")
        with open(hparams_path, "r") as f:
            hparams_dict = json.load(f)
        
        hp = TrainHyperparams()
        for key, value in hparams_dict.items():
            if hasattr(hp, key):
                setattr(hp, key, value)
        
        print(f"✅ Loaded {len(hparams_dict)} hyperparameters")
    else:
        print(f"⚠️  No hyperparameters.json found, using Trial 37 defaults")
        hp = TrainHyperparams()
        # Trial 37 best hyperparameters
        hp.lr_gate = 0.000693424310288062
        hp.lr_cross_attn = 5.8039428979633884e-05
        hp.lr_align = 0.00038210272987863897
        hp.lr_session = 0.00019606824039492657
        hp.gate_bias_init = -0.32431152068047325
        hp.gate_temperature_init = 1.0959691973956611
        hp.graph_gate_bias_init = -0.6955597228515471
        hp.graph_gate_temperature_init = 1.5790821511577782
        hp.graph_gate_boost_weight = 0.24392438684045603
        hp.warmup_steps = 50
        hp.max_grad_norm = 1.3392524375506107
        hp.weight_decay = 0.05395115580901312
        hp.session_num_layers = 2
        hp.session_num_heads = 4
    
    print(f"\n🎛️  Key Hyperparameters:")
    print(f"   gate_bias_init: {hp.gate_bias_init:.4f}")
    print(f"   gate_temperature_init: {hp.gate_temperature_init:.4f}")
    print(f"   graph_gate_bias_init: {hp.graph_gate_bias_init:.4f}")
    print(f"   graph_gate_boost_weight: {hp.graph_gate_boost_weight:.4f}")
    
    return hp

# Verify checkpoint
if VERIFY_MODEL_STATE:
    trainer_state = verify_checkpoint_state(CHECKPOINT_PATH)

hyperparams = load_hyperparams_from_checkpoint(CHECKPOINT_PATH)

# -------------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------------
print(f"📦 Loading models...")

# Base T5
llm_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, legacy=False)
baseline_llm_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
baseline_llm_model.to(DEVICE).eval()
print(f"✅ Loaded {BASE_MODEL_NAME}")

# BGE embedder
emb_tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
bge_model = AutoModel.from_pretrained(BGE_MODEL_NAME)
bge_model.to(DEVICE).eval()
emb_dim = bge_model.config.hidden_size
print(f"✅ Loaded {BGE_MODEL_NAME} (dim={emb_dim})")

# StageAB model
personal_model = PersonalizedModel(
    llm=baseline_llm_model,
    emb_dim=emb_dim,
    max_his_len=MAX_HIS_LEN,
    hyperparams=hyperparams,
    device=DEVICE
)

# Load checkpoint
checkpoint_file = os.path.join(CHECKPOINT_PATH, "model.safetensors")
if not os.path.exists(checkpoint_file):
    checkpoint_file = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")

state_dict = torch.load(checkpoint_file, map_location=DEVICE, weights_only=True)

print(f"\n📊 Loaded state dict: {len(state_dict)} parameters")

# Print important gate parameters
important_params = [
    'gate_network.bias',
    'gate_network.temperature',
    'graph_gate_network.bias',
    'graph_gate_network.temperature',
]

print(f"\n🔍 Important parameter values:")
for param_name in important_params:
    if param_name in state_dict:
        value = state_dict[param_name]
        if value.numel() == 1:
            print(f"  {param_name}: {value.item():.6f}")

missing, unexpected = personal_model.load_state_dict(state_dict, strict=False)

if missing:
    print(f"\n⚠️  Missing keys: {len(missing)}")
    for key in missing[:10]:  # Show first 10
        print(f"     - {key}")
if unexpected:
    print(f"\n⚠️  Unexpected keys: {len(unexpected)}")
    for key in unexpected[:10]:  # Show first 10
        print(f"     - {key}")

personal_model.to(DEVICE).eval()
print(f"✅ Loaded personalized model from checkpoint")

# Load embeddings
emb_path = f"../bge_emb/task_{TASK_ID}_train_bge.npy"
offline_embs = np.load(emb_path, mmap_mode='r')
print(f"✅ Loaded embeddings: shape={offline_embs.shape}")

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def extract_numeric_rating(text: str) -> Optional[int]:
    """Extract numeric rating from text."""
    if text is None:
        return None
    text = str(text).strip()
    if text in ["1", "2", "3", "4", "5"]:
        return int(text)
    match = re.search(r'\b([1-5])\b', text)
    return int(match.group(1)) if match else None

def rating_probs_from_first_step(logits_1step: torch.Tensor) -> Dict[str, Any]:
    """Extract rating probabilities from logits."""
    if logits_1step.dim() == 2:
        logits_1step = logits_1step[0]

    digit_ids = torch.tensor(
        [_DIGIT_TOKEN_IDS[str(i)] for i in range(1, 6)],
        device=logits_1step.device,
        dtype=torch.long
    )
    digit_logits = logits_1step.index_select(dim=0, index=digit_ids)
    probs = F.softmax(digit_logits, dim=-1)

    vals = probs.detach().cpu().tolist()
    items = {str(i + 1): float(vals[i]) for i in range(5)}
    
    top2 = sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:2]
    top_choice, top_prob = top2[0]
    second_prob = top2[1][1] if len(top2) > 1 else 0.0

    entropy = float(-(probs * (probs + 1e-12).log()).sum().detach().cpu().item())
    margin = float(top_prob - second_prob)
    
    return {
        "p": items,
        "top": top_choice,
        "top_prob": float(top_prob),
        "margin": margin,
        "entropy": entropy
    }

def run_llm_base_with_transparency(prompt: str) -> Tuple[str, str, Dict[str, Any]]:
    """Run baseline Flan-T5."""
    tokens = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    
    gen = baseline_llm_model.generate(
        **tokens,
        num_beams=NUM_BEAMS,
        max_new_tokens=2,
        min_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    first_step_logits = gen.scores[0]
    info = rating_probs_from_first_step(first_step_logits)
    text = llm_tokenizer.decode(gen.sequences[0], skip_special_tokens=True).strip()
    
    return info["top"], text, info

def run_stageab_forward_1step(
    prompt: str,
    his_id: torch.Tensor,
    graph_node_ids: Optional[torch.Tensor],
    return_diagnostics: bool = True
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Run StageAB forward with gate diagnostics."""
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    dummy_labels = torch.full_like(llm_inp["input_ids"], fill_value=-100)

    graph_node_mask = None
    if graph_node_ids is not None:
        graph_node_mask = graph_node_ids.ge(0).long()

    # Clear gate tracking
    if hasattr(personal_model, '_gate_means'):
        personal_model._gate_means = []
    if hasattr(personal_model, '_graph_gate_means'):
        personal_model._graph_gate_means = []

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
    if logits is None:
        return "(no_logits)", "", None

    logits_1step = logits[:, 0, :]
    info = rating_probs_from_first_step(logits_1step[0])
    rating = info["top"]

    # Gate diagnostics
    if return_diagnostics and hasattr(personal_model, '_gate_means') and personal_model._gate_means:
        gate_mean = float(np.mean(personal_model._gate_means))
        info["gate_mean"] = gate_mean
        
        if gate_mean > GATE_VERY_OPEN_THRESHOLD:
            info["gate_status"] = "🔴 VERY OPEN"
        elif gate_mean < GATE_VERY_CLOSED_THRESHOLD:
            info["gate_status"] = "🔵 VERY CLOSED"
        else:
            info["gate_status"] = "🟢 BALANCED"
    
    # Graph gate diagnostics
    if return_diagnostics and hasattr(personal_model, '_graph_gate_means') and personal_model._graph_gate_means:
        graph_gate_mean = float(np.mean(personal_model._graph_gate_means))
        info["graph_gate_mean"] = graph_gate_mean
        effective_contrib = graph_gate_mean * hyperparams.graph_gate_boost_weight
        info["graph_effective_contrib"] = effective_contrib

    return rating, rating, info

def print_prediction_with_diagnostics(label: str, rating: str, text: str, info: Optional[Dict]):
    """Print prediction with diagnostics."""
    print(f"  {label:30s} → {text}")
    
    if info is None:
        return
    
    probs_str = " ".join([f"{k}:{info['p'][k]:.2f}" for k in sorted(info['p'].keys())])
    print(f"  [Probs: {probs_str}]")
    print(f"  [Margin={info['margin']:.3f} | Entropy={info['entropy']:.3f}]")
    
    if 'gate_mean' in info:
        status = info.get('gate_status', '')
        print(f"  {status} [Gate={info['gate_mean']:.4f}]")
    
    if 'graph_effective_contrib' in info:
        print(f"  [Graph Contrib={info['graph_effective_contrib']:.4f}]")

def _metric_block_enhanced(name: str, get_pred, rows: List[Dict]):
    """Compute metrics."""
    gold_nums = []
    pred_nums = []
    
    for r in rows:
        if r["gold"] is None:
            continue
        g = extract_numeric_rating(r["gold"])
        p = extract_numeric_rating(get_pred(r))
        
        if g is not None and p is not None:
            gold_nums.append(g)
            pred_nums.append(p)

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
    print(f"  Accuracy:  {acc:.3f} ({acc*100:.1f}%)")
    print(f"  RMSE:      {rmse:.3f}")
    print(f"  MAE:       {mae:.3f}")

# -------------------------------------------------------------------------
# Load test data
# -------------------------------------------------------------------------
offsets_path = f"../bge_emb/task_{TASK_ID}_train_offsets.json"
with open(offsets_path, "r") as f:
    offsets = json.load(f)

candidates = [e for e in offsets.get("entries", []) if len(e.get("profile_id", [])) >= MAX_HIS_LEN]
user_entry = random.choice(candidates)
start_idx = int(user_entry.get("start", 0))

his_row_ids = list(range(start_idx, start_idx + MAX_HIS_LEN))
profile_his_ids = [str(x) for x in user_entry.get("profile_id", [])[:MAX_HIS_LEN]]

his_id_real = torch.tensor([his_row_ids], dtype=torch.long, device=DEVICE)
his_id_empty = torch.zeros_like(his_id_real)

print(f"✅ Using history IDs: {his_row_ids}")

# Graph mapping
his_to_graph_path = f"../graph_emb/task_{TASK_ID}_his_to_graph.json"
review_node_indices = []

if os.path.exists(his_to_graph_path):
    with open(his_to_graph_path, "r") as f:
        node_maps = json.load(f)
    
    for hid in profile_his_ids:
        node_idx = int(node_maps.get(hid, node_maps.get(f"review_{hid}", -1)))
        review_node_indices.append(node_idx)
    
    print(f"✅ Mapped {sum(1 for x in review_node_indices if x >= 0)}/{len(profile_his_ids)} to graph nodes")
else:
    review_node_indices = [-1] * MAX_HIS_LEN

# Load dev questions
if DEV_SUBSET:
    questions_path = f"../LaMP_time_{TASK_ID}_subset/dev_questions.json"
    outputs_path = f"../LaMP_time_{TASK_ID}_subset/dev_outputs.json"
else:
    questions_path = f"../LaMP_time_{TASK_ID}/dev_questions.json"
    outputs_path = f"../LaMP_time_{TASK_ID}/dev_outputs.json"

with open(questions_path, "r") as fq:
    questions_data = json.load(fq)
with open(outputs_path, "r") as fo:
    outputs_data = json.load(fo)

id_to_gold = {g["id"]: str(g["output"]).strip() for g in outputs_data.get("golds", [])}
candidates_q = [q for q in questions_data if q.get("id") in id_to_gold]
sampled = random.sample(candidates_q, k=min(DEV_SAMPLE_SIZE, len(candidates_q)))

test_items = [
    {"id": q.get("id"), "prompt": q.get("input", ""), "gold": id_to_gold.get(q.get("id"), "")}
    for q in sampled
]

# -------------------------------------------------------------------------
# Run comparison
# -------------------------------------------------------------------------
rows = []

for item in test_items:
    prompt = item["prompt"]
    gold = item["gold"]
    qid = item["id"]

    print(f"\n{'='*80}")
    print(f"🧾 ID: {qid} | GOLD: {gold}")
    print(f"{'='*80}")

    out_base = out_pers = out_emp = out_graph = None
    pers_info = graph_info = None

    if SHOW_BASE:
        base_rating, base_text, base_info = run_llm_base_with_transparency(prompt)
        print_prediction_with_diagnostics("BASE", base_rating, base_text, base_info)
        out_base = base_rating

    if SHOW_PERSONA_REAL:
        pers_rating, pers_text, pers_info = run_stageab_forward_1step(prompt, his_id_real, None)
        print_prediction_with_diagnostics("PERSONA (REAL)", pers_rating, pers_text, pers_info)
        out_pers = pers_rating

    if SHOW_GRAPH:
        graph_node_ids_tensor = torch.tensor([review_node_indices], dtype=torch.long, device=DEVICE)
        graph_rating, graph_text, graph_info = run_stageab_forward_1step(prompt, his_id_real, graph_node_ids_tensor)
        print_prediction_with_diagnostics("PERSONA + GRAPH", graph_rating, graph_text, graph_info)
        out_graph = graph_rating

    rows.append({
        "id": qid,
        "gold": gold,
        "base": out_base,
        "pers": out_pers,
        "graph": out_graph,
        "pers_info": pers_info,
        "graph_info": graph_info,
    })

# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
if SHOW_BASE:
    _metric_block_enhanced("BASE (Flan-T5-Large)", lambda r: r["base"], rows)
if SHOW_PERSONA_REAL:
    _metric_block_enhanced("PERSONA (REAL)", lambda r: r["pers"], rows)
if SHOW_GRAPH:
    _metric_block_enhanced("PERSONA + GRAPH", lambda r: r["graph"], rows)

print(f"\n{'='*80}")
print(f"✅ Comparison complete!")
print(f"{'='*80}\n")