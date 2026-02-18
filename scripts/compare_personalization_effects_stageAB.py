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
CHECKPOINT_PATH = "../extention/output_3/checkpoint-86"  # <-- change to your StageAB checkpoint folder
MAX_HIS_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def run_stageab_forward_1step(prompt: str, his_id: torch.Tensor, graph_node_ids: Optional[torch.Tensor]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Run StageAB forward() once and interpret logits for the first decoded position.
    We create labels filled with -100 (ignore) so forward() runs without training signal.
    """
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    dummy_labels = torch.full_like(llm_inp["input_ids"], fill_value=-100)

    graph_node_mask = None
    if graph_node_ids is not None:
        graph_node_mask = graph_node_ids.ge(0).long()

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

    # logits: (B, T, V) -> take first decoded position
    logits_1step = logits[:, 0, :]  # (B, V)
    info = rating_probs_from_first_step(logits_1step[0])
    rating = info["top"]

    # Since StageAB forward is teacher-forcing without generate(), we don't have a natural decoded sequence here.
    # Return just the predicted rating as "text".
    return rating, rating, info

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
his_id_real = torch.tensor([his_row_ids], dtype=torch.long, device=DEVICE)
his_id_empty = torch.zeros_like(his_id_real)

print(f"✅ Using offsets start_idx={start_idx} -> his_row_ids={his_row_ids}")

# -------------------------------------------------------------------------
# Optional: graph id mapping (expects the same files your other script uses)
# -------------------------------------------------------------------------
review_node_indices: List[int] = [-1] * MAX_HIS_LEN
his_to_graph_path = f"../graph_emb/task_{TASK_ID}_his_to_graph.json"

if os.path.exists(his_to_graph_path):
    try:
        with open(his_to_graph_path, "r") as f:
            node_maps = json.load(f)

        # We do NOT have original his_ids here, only memmap row ids.
        # If your mapping is keyed by the original review id, this may not align.
        # For now, leave indices as -1 unless your node_maps supports row-id lookup.
        if isinstance(node_maps, dict):
            for i, rid in enumerate(his_row_ids):
                k = str(rid)
                if k in node_maps:
                    review_node_indices[i] = int(node_maps[k])
    except Exception:
        pass

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

    print(f"\n🧾 ID: {qid} | Prompt: {prompt}")
    if gold is not None:
        print(f"  GOLD                        → {gold}")

    base_info = pers_info = emp_info = graph_info = None
    out_base = out_pers = out_emp = out_graph = None

    if SHOW_BASE:
        base_rating, base_text, base_info = run_llm_base_with_transparency(prompt)
        print(f"  BASE (Flan‑T5)               → {base_text}")
        print(f"  [BASE rating/probs]          → rating={base_rating} p={base_info['p']} margin={base_info['margin']:.3f} H={base_info['entropy']:.3f}")
        out_base = base_rating

    if SHOW_PERSONA_REAL:
        pers_rating, pers_text, pers_info = run_stageab_forward_1step(prompt, his_id_real, graph_node_ids=None)
        print(f"  StageAB (REAL)               → {pers_text}")
        if pers_info is not None:
            print(f"  [StageAB rating/probs]       → rating={pers_rating} p={pers_info['p']} margin={pers_info['margin']:.3f} H={pers_info['entropy']:.3f}")
        out_pers = pers_rating

    if SHOW_PERSONA_EMPTY:
        emp_rating, emp_text, emp_info = run_stageab_forward_1step(prompt, his_id_empty, graph_node_ids=None)
        print(f"  StageAB (EMPTY)              → {emp_text}")
        if emp_info is not None:
            print(f"  [EMPTY rating/probs]         → rating={emp_rating} p={emp_info['p']} margin={emp_info['margin']:.3f} H={emp_info['entropy']:.3f}")
        out_emp = emp_rating

    if SHOW_GRAPH:
        # If your graph_node_ids are all -1, this path is effectively "no graph".
        graph_rating, graph_text, graph_info = run_stageab_forward_1step(prompt, his_id_real, graph_node_ids=graph_node_ids)
        print(f"  StageAB (REAL + GRAPH)       → {graph_text}")
        if graph_info is not None:
            print(f"  [GRAPH rating/probs]         → rating={graph_rating} p={graph_info['p']} margin={graph_info['margin']:.3f} H={graph_info['entropy']:.3f}")
        out_graph = graph_rating

    rows.append({
        "id": qid,
        "gold": gold,
        "base": out_base,
        "pers": out_pers,
        "empty": out_emp,
        "graph": out_graph,
    })

# -------------------------------------------------------------------------
# Metrics (numeric match + RMSE/MAE)
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    def _metric_block(name: str, get_pred):
        gold_nums: List[int] = []
        pred_nums: List[int] = []
        for r in rows:
            if r["gold"] is None:
                continue
            g = extract_numeric_rating(r["gold"])
            p = extract_numeric_rating(get_pred(r))
            if g is not None and p is not None:
                gold_nums.append(g)
                pred_nums.append(p)

        print(f"\n=== METRICS: {name} ===")
        if not gold_nums:
            print("  (no parsable numeric pairs)")
            return

        acc = float(np.mean([int(g == p) for g, p in zip(gold_nums, pred_nums)]))
        rmse = float(mean_squared_error(gold_nums, pred_nums, squared=False))
        mae = float(mean_absolute_error(gold_nums, pred_nums))
        print(f"  n={len(gold_nums)} | acc={acc:.3f} | rmse={rmse:.3f} | mae={mae:.3f}")

    if SHOW_BASE:
        _metric_block("BASE", lambda r: r["base"])
    if SHOW_PERSONA_REAL:
        _metric_block("StageAB (REAL)", lambda r: r["pers"])
    if SHOW_PERSONA_EMPTY:
        _metric_block("StageAB (EMPTY)", lambda r: r["empty"])
    if SHOW_GRAPH:
        _metric_block("StageAB (REAL + GRAPH)", lambda r: r["graph"])