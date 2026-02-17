"""
Compare outputs of:
  (A) Flan‑T5 baseline (no personalization)
  (B) Personalized model using user profile embeddings
  (C) Personalized + Graph‑enhanced model (profile + GNN embeddings)

Purpose:
  Demonstrate how personalization and graph context modify text generation.
  After fine‑tuning, this script allows side‑by‑side qualitative comparison and
  gate monitoring (personalization strength diagnostics).
"""

import os, sys, json, math, torch
import random
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
import torch.nn.functional as F
torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
# ✅ Setup imports
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extention.ModelForPer_slim_GNN import PersonalLLM_Slim
try:
    from extention.graph_encoder_module import GraphEncoder
except ImportError:
    GraphEncoder = None

# -------------------------------------------------------------------------
# 🔧 CONFIG
# -------------------------------------------------------------------------
TASK_ID = 3
CHECKPOINT_PATH = "../extention/output_3/checkpoint-86"
MAX_HIS_LEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# ✅ USER POOL (optional): restrict demo to these user_ids if present in offsets
# -------------------------------------------------------------------------
INCLUDE_USER_IDS = {
    "9019238", "9013448", "9018434", "9019051", "9011936", "901824", "902727", "901473",
    "9013760", "9017855", "9012246", "9016197", "904016", "906891", "904577", "9015721",
    "905033", "9017209", "904890", "9019000", "902462", "9014709", "905941", "9017979",
    "9012870", "902866", "905922", "9017157", "909958", "901630", "905402", "9016824",
    "901928", "9011246", "9017061", "9013776", "9014951", "9011605", "9014828", "9012202",
    "908498", "905173", "90720", "907379", "909031", "9010309", "905561", "90502",
    "901353", "9015176",
}

# -------------------------------------------------------------------------
# 🔧 DEMO MODES (what to print)
# -------------------------------------------------------------------------
SHOW_BASE = True
SHOW_PERSONA_REAL = True
SHOW_PERSONA_EMPTY = True
SHOW_GRAPH = True   


# -------------------------------------------------------------------------
# 🧪 PERSONALIZATION DEMO OVERRIDES (for qualitative inspection)
# -------------------------------------------------------------------------
FORCE_PERS_DEMO = True          # master switch
FORCE_GATE_VALUE = 0.98         # 0..1 (use ~0.95-1.0 to emphasize persona)
PERS_SCALE = 3.0               # multiplies persona contribution when available
GEN_TEMPERATURE = 0.7           # <1 amplifies differences; try 0.5-0.9
GEN_TOP_P = 1.0                 # keep 1.0 for deterministic-ish; reduce for diversity
GEN_NUM_BEAMS = 1               # beams can wash out tiny differences; try 1 or 3   


# -------------------------------------------------------------------------
# 🔧 DEV GOLD EVAL (optional)
# -------------------------------------------------------------------------
USE_DEV_GOLD = True
DEV_SUBSET = True
DEV_SAMPLE_SIZE = 400

# -------------------------------------------------------------------------
# 🧠 LOAD MODELS
# -------------------------------------------------------------------------
llm_model_path = "../FlanT5-base"
emb_model_path = "../bge-base-en-v1.5"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path, use_fast=False)
emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_path)

# ✅ Baseline T5 (never used inside PersonalLLM_Slim)
baseline_llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_path).to(DEVICE).eval()

# ✅ Separate T5 instance for the personalized wrapper (checkpoint loads here)
pers_llm_model = T5ForConditionalGeneration.from_pretrained(llm_model_path).to(DEVICE).eval()

emb_model = AutoModel.from_pretrained(emb_model_path)

# -------------------------------------------------------------------------
# 🚀 LOAD PERSONALIZED MODEL (FIXED)
# -------------------------------------------------------------------------
personal_model = PersonalLLM_Slim(
    llm_model=pers_llm_model,
    emb_model=emb_model,
    max_input_len=256,
    max_new_len=64,
    task_id=TASK_ID
).to(DEVICE)
personal_model.llm_tokenizer = llm_tokenizer

# ✅ FIX: Resize model embeddings to match checkpoint vocabulary
# This must happen BEFORE loading the checkpoint
checkpoint_vocab_size = 32100  # From error message
current_vocab_size = personal_model.llm_model.shared.weight.shape[0]

if current_vocab_size != checkpoint_vocab_size:
    print(f"⚠️  Vocab size mismatch: current={current_vocab_size}, checkpoint={checkpoint_vocab_size}")
    print(f"   Resizing model embeddings to {checkpoint_vocab_size}...")
    personal_model.llm_model.resize_token_embeddings(checkpoint_vocab_size)
    print(f"✅ Model resized to match checkpoint")

# ✅ CRITICAL FIX: Load trained checkpoint
if os.path.exists(CHECKPOINT_PATH):
    print(f"📦 Loading checkpoint from {CHECKPOINT_PATH}")
    
    # Try different checkpoint file names
    for ckpt_file in ["pytorch_model.bin", "model.pt", "checkpoint.pt"]:
        ckpt_path = os.path.join(CHECKPOINT_PATH, ckpt_file)
        if os.path.exists(ckpt_path):
            print(f"  Found: {ckpt_file}")
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            
            # Handle trainer_state format
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                checkpoint = checkpoint['model']
            
            missing, unexpected = personal_model.load_state_dict(checkpoint, strict=False)
            
            if missing:
                print(f"  ⚠️  Missing keys: {len(missing)} (showing first 5)")
                for key in missing[:5]:
                    print(f"    - {key}")
            if unexpected:
                print(f"  ⚠️  Unexpected keys: {len(unexpected)}")
            
            print("  ✅ Weights loaded")
            break
    else:
        raise FileNotFoundError(f"No checkpoint file found in {CHECKPOINT_PATH}")
    
    # Verify critical components loaded
    if hasattr(personal_model, 'gate'):
        # gate is a Sequential, so extract weights from its layers
        gate_weights = []
        for layer in personal_model.gate:
            if hasattr(layer, 'weight'):
                gate_weights.append(torch.norm(layer.weight).item())
        
        if gate_weights:
            gate_norm = sum(gate_weights) / len(gate_weights)  # Average norm
            print(f"  🔍 Gate avg weight norm: {gate_norm:.6f}")
            if gate_norm < 0.01:
                print("  ⚠️  WARNING: Gate weights near zero!")
        else:
            print("  ⚠️  Gate module has no weight parameters")
    
    if hasattr(personal_model, 'align_mlp'):
        mlp_weights = []
        for param in personal_model.align_mlp.parameters():
            mlp_weights.append(torch.norm(param).item())
        
        if mlp_weights:
            mlp_norm = sum(mlp_weights) / len(mlp_weights)
            print(f"  🔍 MLP avg weight norm: {mlp_norm:.6f}")
        else:
            print("  ⚠️  MLP module has no parameters")
else:
    raise FileNotFoundError(f"❌ Checkpoint directory not found: {CHECKPOINT_PATH}")

personal_model.eval()
print("✅ Model ready for inference\n")

print("BASE and PERS share same llm_model object?:", baseline_llm_model is personal_model.llm_model)

# ✅ PATCH: Fix gate dimension mismatch
if hasattr(personal_model, 'gate'):
    gate_first_layer = None
    for layer in personal_model.gate:
        if hasattr(layer, 'in_features'):
            gate_first_layer = layer
            break

    if gate_first_layer is not None:
        expected_in = int(gate_first_layer.in_features)
        print(f"\nℹ️ Gate expects input dim: {expected_in}")
        print("   Installing gate input pad/truncate wrapper...")

        original_forward = personal_model.gate.forward

        def wrapped_gate_forward(x):
            d = int(x.shape[-1])

            if d == expected_in:
                return original_forward(x)

            if d < expected_in:
                pad = x.new_zeros((x.shape[0], expected_in - d))
                x = torch.cat([x, pad], dim=-1)
                return original_forward(x)

            x = x[:, :expected_in]
            return original_forward(x)

        personal_model.gate.forward = wrapped_gate_forward
        print("✅ Gate wrapper installed")

        # ✅ NEW: optionally force gate value (demo only)
        if FORCE_PERS_DEMO:
            _orig_gate_forward_2 = personal_model.gate.forward

            def _forced_gate_forward(x):
                out = _orig_gate_forward_2(x)
                forced = torch.full_like(out, float(FORCE_GATE_VALUE))
                return forced

            personal_model.gate.forward = _forced_gate_forward
            print(f"✅ FORCE_PERS_DEMO enabled: gate forced to ~{FORCE_GATE_VALUE}")

# -------------------------------------------------------------------------
# 🌐 GRAPH ENCODER (optional)
# -------------------------------------------------------------------------
if GraphEncoder:
    try:
        graph_encoder = GraphEncoder(
            num_nodes=5000, emb_dim=768, hidden_dim=256,
            alpha=0.2, dropout=0.3, use_norm=True
        ).to(DEVICE).eval()
    except Exception:
        graph_encoder = None
        print("⚠️ GraphEncoder init failed; skipping graph model.")
else:
    graph_encoder = None
    print("⚠️ GraphEncoder module unavailable.")

# -------------------------------------------------------------------------
# 📑 Example user & review IDs (confirmed valid IDs)
# -------------------------------------------------------------------------
# Automatically pick a valid user and profile_his_ids from offsets file
with open("../bge_emb/task_3_train_offsets.json") as f:
    offsets = json.load(f)

# Pick a user entry (avoid always using the first one with start=0)
min_len = MAX_HIS_LEN

candidates = [
    e for e in offsets.get("entries", [])
    if len(e.get("profile_id", [])) >= min_len
]

# ✅ NEW: filter candidates by INCLUDE_USER_IDS (string-normalized), if possible
if INCLUDE_USER_IDS:
    filtered = []
    for e in candidates:
        pids = e.get("profile_id", [])
        if not pids:
            continue
        uid = str(pids[0])
        if uid in INCLUDE_USER_IDS:
            filtered.append(e)

    if filtered:
        candidates = filtered
        print(f"✅ Restricting offsets candidates to INCLUDE_USER_IDS: {len(candidates)} matches found")
    else:
        print("⚠️ None of INCLUDE_USER_IDS were found in offsets; falling back to random valid user")

if not candidates:
    raise ValueError(f"No offsets entry has at least MAX_HIS_LEN={min_len} profiles.")

# Optional: avoid the very first block (start=0) unless it's the only choice
candidates_nonzero = [e for e in candidates if int(e.get("start", 0)) > 0]
user_entry = random.choice(candidates_nonzero or candidates)

start_idx = int(user_entry["start"])
print(f"✅ Picked offsets entry with start_idx={start_idx}, n_profiles={user_entry.get('n_profiles')}, "
      f"history_len={len(user_entry.get('profile_id', []))}")

user_id = user_entry["profile_id"][0]  # Use the first profile_id as user_id (or use question_index if needed)
profile_his_ids = user_entry["profile_id"][:MAX_HIS_LEN]  # Truncate to MAX_HIS_LEN

print(f"✅ Picked user_id: {user_id}")
print(f"✅ Picked profile_his_ids: {profile_his_ids}")

# -------------------------------------------------------------------------
# 🔄 Load mapping files safely
# -------------------------------------------------------------------------
try:
    with open("../graph_emb/task_3_his_to_graph.json") as f:
        node_maps = json.load(f)
    with open("../graph_emb/task_3_processed_ids.json") as f:
        id_to_index = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("❌ Missing graph files. Run embedding/graph pipeline first.")

# Determine ID table shape
id_map_is_list = isinstance(id_to_index, list)
if id_map_is_list:
    print("ℹ️ task_3_processed_ids.json is a list — treating entries as sequential indices.")

# --- PATCH: Robust review node index extraction ---
review_node_indices = []
for hid in profile_his_ids:
    idx = None
    # Try direct integer index if mapping is a list
    if isinstance(node_maps, list):
        try:
            idx = int(hid)
            if 0 <= idx < len(node_maps):
                review_node_indices.append(idx)
            else:
                review_node_indices.append(-1)
        except Exception:
            review_node_indices.append(-1)
    # Try dictionary lookup if mapping is a dict
    elif isinstance(node_maps, dict):
        idx = node_maps.get(hid) or node_maps.get(f"review_{hid}") or node_maps.get(str(hid))
        review_node_indices.append(idx if idx is not None else -1)
    else:
        review_node_indices.append(-1)
print(f"🔗 Review node indices for user {user_id}: {review_node_indices}")

# -------------------------------------------------------------------------
# 🧩 Build valid history tensor
# -------------------------------------------------------------------------
memmap_size = getattr(personal_model, "his_train_memmap", torch.zeros((1,))).shape[0]

# Build memmap row indices for history
his_row_ids = list(range(start_idx, start_idx + MAX_HIS_LEN))
his_id_real = torch.tensor([his_row_ids], dtype=torch.long, device=DEVICE)

his_id_empty = torch.zeros_like(his_id_real)

print("\n🔍 PERSONA EMBEDDING DIAGNOSTICS")
print(f"his_id_real shape: {his_id_real.shape}")
print(f"his_id_real values: {his_id_real[0].tolist()}")
print(f"Number of non-zero IDs: {(his_id_real != 0).sum().item()}")

# Test if profile embedding differs from baseline
with torch.no_grad():
    # Get task embedding (same for both)
    dummy_tokens = emb_tokenizer("test", return_tensors="pt").to(DEVICE)
    
    # ✅ FIX: Compute task embedding directly from the embedding model
    # instead of calling a non-existent method
    emb_model_output = personal_model.emb_model(
        input_ids=dummy_tokens["input_ids"],
        attention_mask=dummy_tokens["attention_mask"],
    )
    # For BGE embeddings, use the pooled output (CLS token)
    task_emb = emb_model_output.last_hidden_state[:, 0, :]  # (1, 768)
    
    # Compare real vs empty persona
    if hasattr(personal_model, 'obtain_profile_emb'):
        # If method exists, use it
        profile_real = personal_model.obtain_profile_emb(his_id_real, task_emb)
        profile_empty = personal_model.obtain_profile_emb(his_id_empty, task_emb)
    else:
        # ✅ Fallback: Compute profile embedding manually
        # Fetch embeddings from memmap and fuse with task embedding
        memmap = personal_model.his_train_memmap if hasattr(personal_model, 'his_train_memmap') else None
        
        if memmap is not None and his_id_real[0, 0].item() > 0:
            # Get real profile embeddings
            valid_ids = his_id_real[0][his_id_real[0] > 0].cpu().numpy()
            real_embs = torch.from_numpy(np.array([memmap[int(idx)] for idx in valid_ids])).to(DEVICE).float()
            profile_real = real_embs.mean(dim=0, keepdim=True)  # (1, 768)
        else:
            profile_real = torch.zeros_like(task_emb)
        
        # Empty profile is always zeros
        profile_empty = torch.zeros_like(task_emb)
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(profile_real, profile_empty).item()
    
    print(f"\n📊 Profile embedding comparison:")
    print(f"  Real profile norm:  {torch.norm(profile_real).item():.4f}")
    print(f"  Empty profile norm: {torch.norm(profile_empty).item():.4f}")
    print(f"  Cosine similarity:  {cos_sim:.4f}")
    
    if cos_sim > 0.99:
        print("  ⚠️ WARNING: Real and empty profiles are nearly identical!")
        print("     → Profile IDs may not be mapping correctly")
    elif cos_sim < 0.5:
        print("  ✅ Real profile is distinct from empty (good!)")
    else:
        print("  ⚠️ Profiles somewhat similar but not identical")

    # Check if gate is actually being applied
    if hasattr(personal_model, 'gate'):
        # ✅ FIX: Inspect gate architecture to determine input dimension
        gate_input_dim = None
        for layer in personal_model.gate:
            if hasattr(layer, 'in_features'):
                gate_input_dim = layer.in_features
                break
        
        if gate_input_dim is None:
            print("  ⚠️ Could not determine gate input dimension; skipping gate test")
        else:
            print(f"  ℹ️ Gate expects input dim: {gate_input_dim}")
            
            # If gate expects more dims, pad or use only task embedding
            if gate_input_dim == 768:
                # Gate takes only task embedding
                gate_input = task_emb
            elif gate_input_dim == 1536:
                # Gate takes concatenation of task + profile (normal case)
                gate_input = torch.cat([task_emb, profile_real], dim=-1)
            elif gate_input_dim == 3072:
                # ✅ Gate expects 4x concatenation (e.g., [task, profile, task, profile])
                # This sometimes happens if gate was trained with different fusion strategy
                gate_input = torch.cat([task_emb, profile_real, task_emb, profile_real], dim=-1)
            else:
                # Try to infer: assume gate wants concatenation of some multiple
                concat_factor = max(1, gate_input_dim // 768)
                print(f"  ℹ️ Inferring concat factor: {concat_factor}")
                embeddings_to_concat = [task_emb, profile_real] * (concat_factor // 2)
                if len(embeddings_to_concat) * 768 < gate_input_dim:
                    embeddings_to_concat.append(task_emb)
                gate_input = torch.cat(embeddings_to_concat[:concat_factor], dim=-1)
            
            try:
                # Apply gate (it's a Sequential, so call it directly)
                gate_output = personal_model.gate(gate_input)
                gate_val = torch.sigmoid(gate_output).item() if torch.is_tensor(gate_output) else 0.5
                
                print(f"\n🚪 Gate activation: {gate_val:.4f}")
                if gate_val < 0.1:
                    print("  ⚠️ WARNING: Gate is nearly closed (personalization suppressed)")
                elif gate_val > 0.9:
                    print("  ✅ Gate is open (strong personalization)")
            except RuntimeError as e:
                print(f"  ❌ Gate forward failed: {e}")
                print(f"     (gate input shape: {gate_input.shape})")
    
    # Check memmap integrity
    if hasattr(personal_model, 'his_train_memmap'):
        memmap = personal_model.his_train_memmap
        print(f"\n💾 Memmap info:")
        print(f"  Shape: {memmap.shape}")
        print(f"  Valid ID range: [0, {memmap.shape[0]})")
        
        # Sample a few embeddings to check if they're distinct
        if memmap.shape[0] > 10:
            sample_ids = [1, 5, 10]
            sample_embs = np.array([memmap[idx] for idx in sample_ids if idx < memmap.shape[0]])
            
            if len(sample_embs) > 1:
                # Check if embeddings are all the same (indicates broken memmap)
                diffs = []
                for i in range(len(sample_embs)-1):
                    cos = np.dot(sample_embs[i], sample_embs[i+1]) / (
                        np.linalg.norm(sample_embs[i]) * np.linalg.norm(sample_embs[i+1]) + 1e-8
                    )
                    diffs.append(cos)
                
                print(f"  Sample embedding similarities: {[f'{d:.3f}' for d in diffs]}")
                if all(d > 0.99 for d in diffs):
                    print("  ⚠️ WARNING: All embeddings are identical (memmap may be corrupted)")

# -------------------------------------------------------------------------
# ✨ Tensor NaN cleaner
# -------------------------------------------------------------------------
def _sanitize(t: torch.Tensor):
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

def _assert_graph_ids_ok(graph_node_ids: torch.Tensor, where: str):
    if graph_node_ids is None:
        raise ValueError(f"[{where}] graph_node_ids is None (graph path is not used).")
    if not torch.is_tensor(graph_node_ids):
        raise TypeError(f"[{where}] graph_node_ids must be a torch.Tensor, got {type(graph_node_ids)}")
    if graph_node_ids.numel() == 0:
        raise ValueError(f"[{where}] graph_node_ids is empty (no graph nodes -> no graph signal).")

    expected_device = DEVICE if isinstance(DEVICE, torch.device) else torch.device(str(DEVICE))
    if graph_node_ids.device != expected_device:
        raise ValueError(f"[{where}] graph_node_ids on {graph_node_ids.device}, expected {expected_device}")

    if graph_node_ids.dtype not in (torch.int64, torch.long):
        raise ValueError(f"[{where}] graph_node_ids must be int64/long, got {graph_node_ids.dtype}")

def extract_numeric_rating(text: str):
    m = re.search(r"\b([1-5])(\.0+)?\b", str(text))
    return int(m.group(1)) if m else None

def _rating_token_ids():
    # Token ids for "1".."5" as single tokens in the LLM tokenizer.
    # (Works for SentencePiece tokenizers; we verify single-token encoding.)
    ids = {}
    for d in ["1", "2", "3", "4", "5"]:
        enc = llm_tokenizer.encode(d, add_special_tokens=False)
        ids[d] = enc[0] if len(enc) == 1 else None
    if any(v is None for v in ids.values()):
        bad = [k for k, v in ids.items() if v is None]
        raise ValueError(f"Digits not single-token in tokenizer for: {bad}. "
                         f"Need a different mapping approach.")
    return ids

_DIGIT_TOKEN_IDS = _rating_token_ids()

def rating_probs_from_first_step(logits_1step: torch.Tensor):
    """
    logits_1step: (V,) or (1, V) logits for the FIRST generated token.
    Returns dict with probs for digits 1..5 and diagnostics.
    """
    if logits_1step.dim() == 2:
        logits_1step = logits_1step[0]
    digit_ids = torch.tensor([_DIGIT_TOKEN_IDS[str(i)] for i in range(1, 6)],
                             device=logits_1step.device, dtype=torch.long)
    digit_logits = logits_1step.index_select(dim=0, index=digit_ids)  # (5,)
    probs = F.softmax(digit_logits, dim=-1)  # (5,)

    vals = probs.detach().cpu().tolist()
    items = {str(i+1): float(vals[i]) for i in range(5)}
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
        "entropy": entropy,
    }

def _decode_from_gen_out(gen_out, tokenizer):
    seq = gen_out.sequences
    if seq is None or seq.numel() == 0:
        return ""
    return tokenizer.decode(seq[0], skip_special_tokens=True).strip()

def _first_step_logits_from_gen_out(gen_out):
    # scores is a tuple(list) of length = generated steps; take step 0
    if not hasattr(gen_out, "scores") or gen_out.scores is None or len(gen_out.scores) == 0:
        return None
    return gen_out.scores[0]  # (B, vocab)

def _rating_probs_from_first_step_logits(first_step_logits, tokenizer, device):
    rating_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    rating_ids_t = torch.tensor(rating_ids, dtype=torch.long, device=device)
    rating_logits = first_step_logits.index_select(dim=-1, index=rating_ids_t)  # (B,5)
    probs = torch.softmax(rating_logits, dim=-1)  # (B,5)
    return rating_ids, probs

def _entropy_and_confidence(probs_1x5):
    # probs_1x5: (5,) tensor
    p = probs_1x5.clamp_min(1e-12)
    ent = float(-(p * p.log()).sum().item())
    conf = float(p.max().item())
    return ent, conf

def _max_prob_delta(p_a, p_b):
    def _to_list(x):
        # Accept dict {'1':p1,...,'5':p5} or list/tuple/np/tensor
        if isinstance(x, dict):
            return [float(x[str(i)]) for i in range(1, 6)]
        if torch.is_tensor(x):
            return x.detach().cpu().flatten().tolist()
        if isinstance(x, (list, tuple, np.ndarray)):
            return [float(v) for v in list(x)]
        raise TypeError(f"Unsupported prob container type: {type(x)}")

    a = torch.tensor(_to_list(p_a), dtype=torch.float32)
    b = torch.tensor(_to_list(p_b), dtype=torch.float32)
    return float((a - b).abs().max().item())

def summarize_personal_model_out(out, llm_tokenizer):
    """
    Returns dict with:
      pred_text, pred_rating, probs(list len 5), entropy, confidence
    """
    if not isinstance(out, dict) or "gen_out" not in out:
        return {"pred_text": "", "pred_rating": None, "probs": None, "entropy": None, "confidence": None}

    gen_out = out["gen_out"]
    pred_text = _decode_from_gen_out(gen_out, llm_tokenizer)

    first_step_logits = _first_step_logits_from_gen_out(gen_out)
    if first_step_logits is None:
        return {"pred_text": pred_text, "pred_rating": None, "probs": None, "entropy": None, "confidence": None}

    rating_ids, probs = _rating_probs_from_first_step_logits(first_step_logits, llm_tokenizer, first_step_logits.device)
    p0 = probs[0].detach().cpu()
    pred_idx = int(p0.argmax().item())
    pred_rating = pred_idx + 1
    ent, conf = _entropy_and_confidence(p0)

    return {
        "pred_text": pred_text,
        "pred_rating": pred_rating,
        "probs": [float(x) for x in p0.tolist()],
        "entropy": ent,
        "confidence": conf,
    }

def _seqs_to_text(seqs) -> str:
    """
    PersonalLLM_Slim.forward() may return:
      - token id tensors
      - list[int]
      - list[list[int]] (nested/beam-like)
      - already-decoded strings / list[str]
    This helper normalizes to a string safely.
    """
    if seqs is None:
        return ""

    # Unwrap one level if it's a tuple/list container (e.g., (seqs,) or [seqs])
    obj = seqs

    # If it's a tuple/list, pick the first element (batch item / first beam container)
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        obj = obj[0]

    # If it's already a string
    if isinstance(obj, str):
        return obj.strip()

    # If it's a torch tensor: handle (T,) or (B,T) or (B,beam,T) by taking first indices
    if torch.is_tensor(obj):
        t = obj
        while t.dim() > 1:
            t = t[0]
        ids = t.detach().cpu().tolist()
        return llm_tokenizer.decode(ids, skip_special_tokens=True).strip()

    # If it's a nested python list (e.g., [[1,2,3]] or [[[...]]]), unwrap until 1D
    if isinstance(obj, list):
        x = obj
        while isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            x = x[0]

        # Now x should be list[int] (or empty)
        if len(x) == 0:
            return ""

        if isinstance(x[0], int):
            return llm_tokenizer.decode(x, skip_special_tokens=True).strip()

        # Rare case: list of strings
        if isinstance(x[0], str):
            return str(x[0]).strip()

        # Fallback: stringify
        return str(x).strip()

    # Fallback: stringify anything else
    return str(obj).strip()

def _forward_out_to_text(out) -> str:
    """
    Handles various PersonalLLM_Slim.forward() return formats:
      - {"loss":..., "gen_out": GenerateEncoderDecoderOutput(...)}   ✅
      - (loss, seqs)
      - (loss, logits)
      - {'loss':..., 'logits':...}
    If logits are returned, decodes only the FIRST generated token (rating).
    """
    if out is None:
        return ""

    def _decode_first_from_logits(logits: torch.Tensor) -> str:
        # logits: (B, T, V) -> take first position only
        first_token_id = logits[:, 0, :].argmax(dim=-1)  # (B,)
        return llm_tokenizer.decode(first_token_id[0].detach().cpu().tolist(), skip_special_tokens=True).strip()

    # dict output
    if isinstance(out, dict):
        if "gen_out" in out and out["gen_out"] is not None:
            return _decode_from_gen_out(out["gen_out"], llm_tokenizer)
        if "sequences" in out:
            return _seqs_to_text(out["sequences"])
        if "logits" in out and torch.is_tensor(out["logits"]) and out["logits"].dim() == 3:
            return _decode_first_from_logits(out["logits"])
        return str(out).strip()

    # tuple/list output
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        second = out[1]
        if torch.is_tensor(second) and second.dim() == 3:
            return _decode_first_from_logits(second)
        return _seqs_to_text(second)

    return str(out).strip()

def _extract_first_step_logits_from_forward_out(out):
    """
    Returns (logits_1step, ok_flag).

    Supports:
      - dict with "gen_out": GenerateEncoderDecoderOutput (use gen_out.scores[0])   ✅
      - dict with "logits": (B,T,V)
      - tuple/list where second item is logits: (B,T,V)
    """
    # ✅ NEW: handle PersonalLLM_Slim return format {"loss":..., "gen_out":...}
    if isinstance(out, dict) and "gen_out" in out and out["gen_out"] is not None:
        gen_out = out["gen_out"]
        if hasattr(gen_out, "scores") and gen_out.scores is not None and len(gen_out.scores) > 0:
            return gen_out.scores[0], True  # (B, V) logits for first generated token

    if isinstance(out, dict) and "logits" in out and torch.is_tensor(out["logits"]) and out["logits"].dim() == 3:
        return out["logits"][:, 0, :], True  # (B,V)

    if isinstance(out, (list, tuple)) and len(out) >= 2:
        second = out[1]
        if torch.is_tensor(second) and second.dim() == 3:
            return second[:, 0, :], True

    return None, False

# -------------------------------------------------------------------------
# 🔧 Generation helpers
# -------------------------------------------------------------------------
def run_llm_base(prompt: str) -> str:
    """Generate text using plain Flan‑T5."""
    tokens = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    out = baseline_llm_model.generate(**tokens, num_beams=3, min_new_tokens=8, max_new_tokens=32)
    return llm_tokenizer.decode(out[0], skip_special_tokens=True).strip()

def run_llm_base_with_transparency(prompt: str):
    tokens = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    gen = baseline_llm_model.generate(
        **tokens,
        num_beams=3,
        max_new_tokens=2,
        min_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # scores[0] = logits for first generated token (shape: (B, V))
    first_step_logits = gen.scores[0]  # (1, V)
    info = rating_probs_from_first_step(first_step_logits)

    text = llm_tokenizer.decode(gen.sequences[0], skip_special_tokens=True).strip()
    # Prefer the probability top choice as the "rating" (more robust than parsing messy text)
    rating = info["top"]
    return rating, text, info

def run_personal(prompt: str, his_id: torch.Tensor) -> str:
    """Generate text using PersonalLLM_Slim with persona (go through personal_model.forward)."""
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    # PersonalLLM_Slim.forward() requires labels; use ignore_index everywhere (no training loss).
    dummy_labels = torch.full_like(llm_inp["input_ids"], fill_value=-100)

    with torch.no_grad():
        out = personal_model(
            llm_input_ids=llm_inp["input_ids"],
            llm_attention_mask=llm_inp["attention_mask"],
            labels=dummy_labels,
            emb_input_ids=emb_inp["input_ids"],
            emb_attention_mask=emb_inp["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
            his_id=his_id,
        )

    return _forward_out_to_text(out)

def run_personal_forward_with_transparency(prompt: str, his_id: torch.Tensor, graph_node_ids: torch.Tensor = None, session_ids: torch.Tensor = None):
    """
    Runs the trained PersonalLLM_Slim.forward() path and extracts rating probs from first-step logits (if available).
    """
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    # PersonalLLM_Slim.forward() requires labels; use ignore_index everywhere (no training loss).
    dummy_labels = torch.full_like(llm_inp["input_ids"], fill_value=-100)

    with torch.no_grad():
        out = personal_model(
            llm_input_ids=llm_inp["input_ids"],
            llm_attention_mask=llm_inp["attention_mask"],
            labels=dummy_labels,
            emb_input_ids=emb_inp["input_ids"],
            emb_attention_mask=emb_inp["attention_mask"],
            emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
            his_id=his_id,
            graph_node_ids=graph_node_ids,
            session_ids=session_ids,
        )

    text = _forward_out_to_text(out)

    logits_1step, ok = _extract_first_step_logits_from_forward_out(out)
    if not ok or logits_1step is None:
        rating = extract_numeric_rating(text)
        return (str(rating) if rating is not None else text), text, None

    info = rating_probs_from_first_step(logits_1step[0])
    rating = info["top"]
    return rating, text, info


def run_personal_generate_with_transparency(prompt: str, his_id: torch.Tensor, graph_node_ids: torch.Tensor = None, session_ids: torch.Tensor = None):
    llm_inp = llm_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    emb_inp = emb_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    with torch.no_grad():
        if hasattr(personal_model, "generate") and callable(getattr(personal_model, "generate")):
            gen_out = personal_model.generate(
                llm_input_ids=llm_inp["input_ids"],
                llm_attention_mask=llm_inp["attention_mask"],
                emb_input_ids=emb_inp["input_ids"],
                emb_attention_mask=emb_inp["attention_mask"],
                emb_token_type_ids=torch.zeros_like(emb_inp["input_ids"]),
                his_id=his_id,
                graph_node_ids=graph_node_ids,
                session_ids=session_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=2,
                min_new_tokens=1,
                num_beams=GEN_NUM_BEAMS,
                do_sample=(GEN_TEMPERATURE is not None and GEN_TEMPERATURE != 1.0) or (GEN_TOP_P is not None and GEN_TOP_P < 1.0),
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
            )
            text = _decode_from_gen_out(gen_out, llm_tokenizer)
            first_step_logits = _first_step_logits_from_gen_out(gen_out)
            if first_step_logits is None:
                rating = extract_numeric_rating(text)
                return (str(rating) if rating is not None else text), text, None

            info = rating_probs_from_first_step(first_step_logits[0])
            return info["top"], text, info

        return run_personal_forward_with_transparency(prompt, his_id, graph_node_ids=graph_node_ids, session_ids=session_ids)


def run_personal_with_transparency(prompt: str, his_id: torch.Tensor):
    # ✅ Now uses the model's trained generate (if available) instead of teacher-forcing forward
    return run_personal_generate_with_transparency(prompt, his_id, graph_node_ids=None, session_ids=None)

def run_personal_graph(prompt: str, his_id: torch.Tensor, graph_node_ids: torch.Tensor = None, session_ids: torch.Tensor = None):
    # ✅ Uses trained forward with graph ids
    rating, text, info = run_personal_forward_with_transparency(prompt, his_id, graph_node_ids=graph_node_ids, session_ids=session_ids)
    return text


# -------------------------------------------------------------------------
# 🧪 PROMPTS FOR DEMO / OR DEV SET SAMPLING
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    if DEV_SUBSET:
        questions_path = "../LaMP_time_3_subset/dev_questions.json"
        outputs_path   = "../LaMP_time_3_subset/dev_outputs.json"
    else:
        questions_path = "../LaMP_time_3/dev_questions.json"
        outputs_path   = "../LaMP_time_3/dev_outputs.json"

    if not (os.path.exists(questions_path) and os.path.exists(outputs_path)):
        raise FileNotFoundError(f"Missing dev files: {questions_path} or {outputs_path}")

    with open(questions_path, "r") as fq:
        questions_data = json.load(fq)
    with open(outputs_path, "r") as fo:
        outputs_data = json.load(fo)

    id_to_gold = {g["id"]: str(g["output"]).strip() for g in outputs_data.get("golds", [])}

    # sample entries that have gold
    candidates = [q for q in questions_data if q.get("id") in id_to_gold]
    if not candidates:
        raise ValueError("No dev questions matched gold IDs.")

    sampled = random.sample(candidates, k=min(DEV_SAMPLE_SIZE, len(candidates)))

    # Each item: {"id","prompt","gold"}
    test_items = []
    for q in sampled:
        qid = q.get("id")
        prompt = q.get("input", "")
        gold = id_to_gold.get(qid, "")
        test_items.append({"id": qid, "prompt": prompt, "gold": gold})
else:
    test_items = [{"id": None, "prompt": p, "gold": None} for p in [
        "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The story was fun but predictable.",
        "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: The shipping was fast!",
        "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: This tent survived heavy rain.",
    ]]

# -------------------------------------------------------------------------
# 📊 COMPARISON RUN (+ GOLD)
# -------------------------------------------------------------------------
rows = []

for item in test_items:
    prompt = item["prompt"]
    gold = item["gold"]
    qid = item["id"]

    print(f"\n🧾 ID: {qid} | Prompt: {prompt}")
    if gold is not None:
        print(f"  GOLD                        → {gold}")

    out_base = out_pers = out_emp = out_graph = None
    base_info = pers_info = None

    if SHOW_BASE:
        base_rating, base_text, base_info = run_llm_base_with_transparency(prompt)
        print(f"  BASE (Flan‑T5)               → {base_text}")
        print(f"  [BASE rating/probs]          → rating={base_rating} p={base_info['p']} margin={base_info['margin']:.3f} H={base_info['entropy']:.3f}")
        out_base = base_rating  # store rating for metrics, not raw text

    if SHOW_PERSONA_REAL:
        pers_rating, pers_text, pers_dbg = run_personal_generate_with_transparency(
            prompt, his_id_real, graph_node_ids=None, session_ids=None
        )
        print(f"  PERS (REAL)                  → {pers_text}")
        if pers_dbg is not None:
            print(f"  [PERS rating/probs]          → rating={pers_rating} p={pers_dbg['p']} margin={pers_dbg['margin']:.3f} H={pers_dbg['entropy']:.3f}")
        else:
            print(f"  [PERS rating/probs]          → rating={pers_rating} (no logits available)")

        gate_val = getattr(personal_model, "_last_gate_values", None)
        if gate_val is not None:
            print(f"  [PERS gate runtime]          → mean={float(gate_val.mean().item()):.4f}")

        out_pers = str(pers_rating) if pers_rating is not None else pers_text
        pers_info = pers_dbg

    if SHOW_PERSONA_EMPTY:
        emp_rating, emp_text, emp_dbg = run_personal_generate_with_transparency(
            prompt, his_id_empty, graph_node_ids=None, session_ids=None
        )
        print(f"  PERS (EMPTY)                 → {emp_text}")
        if emp_dbg is not None:
            print(f"  [EMPTY rating/probs]         → rating={emp_rating} p={emp_dbg['p']} margin={emp_dbg['margin']:.3f} H={emp_dbg['entropy']:.3f}")
        else:
            print(f"  [EMPTY rating/probs]         → rating={emp_rating} (no logits available)")

        out_emp = str(emp_rating) if emp_rating is not None else emp_text
        emp_info = emp_dbg

    if SHOW_GRAPH:
        graph_node_ids = torch.tensor([review_node_indices], dtype=torch.long, device=DEVICE)
        graph_node_ids = graph_node_ids.to(DEVICE).long()
        _assert_graph_ids_ok(graph_node_ids, where=f"id={qid}")

        graph_rating, graph_text, graph_info = run_personal_generate_with_transparency(
            prompt, his_id_real, graph_node_ids=graph_node_ids, session_ids=None
        )
        print(f"  Personalized + Graph         → {graph_text}")
        if graph_info is not None:
            print(f"  [GRAPH rating/probs]         → rating={graph_rating} p={graph_info['p']} margin={graph_info['margin']:.3f} H={graph_info['entropy']:.3f} ")
        out_graph = graph_rating

        if graph_info is not None and pers_info is not None:
            print(f"  Δmax(prob) graph-vs-pers: {_max_prob_delta(graph_info['p'], pers_info['p']):.6f}")

    rows.append({
        "id": qid,
        "gold": gold,
        "base": out_base,
        "pers": out_pers,
        "empty": out_emp,
        "graph": out_graph,
        "base_info": base_info,
        "pers_info": pers_info,
        "emp_info": emp_info,
        "graph_info": graph_info,
    })

# -------------------------------------------------------------------------
# 📈 METRICS (numeric match + RMSE/MAE) when gold exists
# -------------------------------------------------------------------------
if USE_DEV_GOLD:
    def _metric_block(name, preds):
        gold_nums = []
        pred_nums = []
        for r in rows:
            if r["gold"] is None:
                continue
            g = extract_numeric_rating(r["gold"])
            p = extract_numeric_rating(preds(r))
            if g is not None and p is not None:
                gold_nums.append(g)
                pred_nums.append(p)

        if not gold_nums:
            print(f"\n=== METRICS: {name} ===")
            print("  (no parsable numeric pairs)")
            return

        acc = float(np.mean([int(g == p) for g, p in zip(gold_nums, pred_nums)]))
        rmse = float(mean_squared_error(gold_nums, pred_nums, squared=False))
        mae = float(mean_absolute_error(gold_nums, pred_nums))
        print(f"\n=== METRICS: {name} ===")
        print(f"  n={len(gold_nums)} | acc={acc:.3f} | rmse={rmse:.3f} | mae={mae:.3f}")

    if SHOW_BASE:
        _metric_block("BASE", lambda r: r["base"])
    if SHOW_PERSONA_REAL:
        _metric_block("PERSONAL (REAL)", lambda r: r["pers"])
    if SHOW_PERSONA_EMPTY:
        _metric_block("PERSONAL (EMPTY)", lambda r: r["empty"])
    if SHOW_GRAPH:
        _metric_block("PERSONAL + GRAPH", lambda r: r["graph"])

# -------------------------------------------------------------------------
# 💡 INTERPRETATION GUIDELINES
# -------------------------------------------------------------------------
# • Differences between BASE and PERSONALIZED show persona influence.
# • gate_mean≈0 → persona ignored; ≈0.5 → balanced; ≈1.0 → strong personalization.
# • Adding Graph context should increase contextual consistency and stability.