
import os
import math
import transformers
from dataclasses import dataclass, field
from typing import Optional

# HuggingFace / Transformers imports:
#  - Tokenizers: build token IDs for Flan-T5 and the embedding model (BGE)
#  - Trainer: orchestrates training/eval loop for seq2seq models
from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    AutoModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

import numpy as np

# TrainerCallback is used for custom console logging and runtime diagnostics.
from transformers import Trainer, TrainerCallback

import torch.nn as nn

# Local project imports (extension architecture):
#  - PersonalDataset emits: LLM tokens, EMB tokens, labels, his_id, session_ids, graph_node_ids/mask
#  - PersonalLLM_Slim wraps Flan-T5 + BGE + (optional) profile/session/graph fusion
from PersonalDataset_profile_GNN import PersonalDataset
from ModelForPer_slim_GNN import PersonalLLM_Slim   # <-- updated slim model
from ModelForPer_slim_GNN_stageA_B import PersonalLLM_Slim_StageAB  # <-- ablation version of slim model

# Eval logging utilities:
#  - extract_rating_1_to_5: parse rating output from generated text
#  - open_jsonl/log_eval_row: optional structured logging for debugging eval samples
from eval_logging import (
        extract_rating_1_to_5, open_jsonl, log_eval_row
    )

from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
import json
from transformers import TrainerCallback
import torch
import re
import argparse

# =============================================================================
# [GENERATION SAFETY] Restrict decoding to a single-token rating {1..5}
# =============================================================================
def _get_allowed_rating_token_ids(tokenizer):
    """
    Build a list of token IDs corresponding to the strings:
        "1", "2", "3", "4", "5"

    Why:
      For LaMP-3 rating prediction, we often want the model to output ONLY a rating.
      During evaluation/generation, we can restrict logits to those tokens to prevent
      verbose answers or unrelated tokens.

    Implementation details:
      - If tokenizer encodes "1" as multiple tokens (rare for T5), we skip it so we
        don't accidentally enforce a broken constraint.
    """
    allowed = []
    for s in ["1", "2", "3", "4", "5"]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            allowed.append(ids[0])
    return allowed


class _RestrictToRatingTokensProcessor(transformers.LogitsProcessor):
    """
    A logits processor that masks ALL tokens except the allowed rating tokens.

    HF generation pipeline calls this on each decoding step:
      - input_ids: current decoded sequence
      - scores: logits for the next token (batch_size, vocab_size)

    We add -inf to all disallowed logits so they never get selected.
    """
    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        # If allowed list is empty (tokenizer splits digits oddly),
        # do nothing to avoid breaking generation.
        if not self.allowed:
            return scores

        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed:
            mask[:, tid] = 0.0
        return scores + mask


class RatingOnlyLogitsProcessorList(transformers.LogitsProcessorList):
    """
    Just a semantic alias so logs/readability clearly indicate our intent.
    """
    pass

# =============================================================================
# [OPTIONAL DIAGNOSTICS] Batch-level checks of personalization inputs
# =============================================================================
def _safe_ratio(x: torch.Tensor) -> float:
    """
    Convert a boolean tensor or numeric tensor to a mean ratio.

    Used to print quick stats like:
      - what fraction of session_ids is non-padding?
      - what fraction of graph ids are valid?

    Returns NaN if tensor is missing/empty.
    """
    if x is None:
        return float("nan")
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.numel() == 0:
        return float("nan")
    return float(x.float().mean().detach().cpu())


class FusionDataDiagnosticsCallback(TrainerCallback):
    """
    Prints lightweight diagnostics about the collated batch so we can verify:
      - session_ids are not mostly padding
      - graph_node_mask has valid nodes
      - graph_node_ids contain non -1 values
      - his_id is present

    Enable with:
      export PPLOG_DIAG=1

    Control frequency:
      export PPLOG_DIAG_EVERY=50

    This is extremely useful when debugging dataset ↔ collator ↔ model alignment.
    """
    def __init__(self):
        super().__init__()
        self.enabled = os.environ.get("PPLOG_DIAG", "").strip().lower() in {"1", "true", "yes", "y"}
        self.every = int(os.environ.get("PPLOG_DIAG_EVERY", "50"))

    def on_train_batch_begin(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        if self.every <= 0:
            return
        if state.global_step % self.every != 0:
            return

        inputs = kwargs.get("inputs", None)
        if inputs is None:
            return

        with torch.no_grad():
            out = {"step": int(state.global_step)}

            # session_ids: padding assumed 0
            if "session_ids" in inputs and inputs["session_ids"] is not None:
                sid = inputs["session_ids"]
                out["session_nonpad_ratio"] = _safe_ratio(sid.ne(0))

            # his_id: padding assumed 0
            if "his_id" in inputs and inputs["his_id"] is not None:
                hid = inputs["his_id"]
                out["his_nonpad_ratio"] = _safe_ratio(hid.ne(0))

            # graph_node_mask: expected 0/1; mean indicates density of valid nodes
            if "graph_node_mask" in inputs and inputs["graph_node_mask"] is not None:
                gmask = inputs["graph_node_mask"]
                out["graph_mask_mean"] = _safe_ratio(gmask)

            # graph_node_ids: padding/invalid assumed -1 (per collator)
            if "graph_node_ids" in inputs and inputs["graph_node_ids"] is not None:
                gids = inputs["graph_node_ids"]
                out["graph_id_valid_ratio"] = _safe_ratio(gids.ne(-1))

            print(f"[DIAG] {json.dumps(out)}")

def str2bool(v):
    """
    Robust bool parser for argparse.
    Accepts: True/False, true/false, 1/0, yes/no, y/n.
    Also supports passing the flag without a value (const=True).
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

def extract_yes_no(text: str):
    """
    Extract a yes/no label from free-form text.

    Why:
      Some custom prompts in this repo expect "yes/no" outputs, so the evaluation
      metrics attempt to handle both:
        - rating (1..5)
        - yes/no classification

    Implementation:
      Use word-boundary regex to avoid matching "yesterday" as "yes", etc.
    """
    if text is None:
        return None
    text_lower = str(text).lower()
    if re.search(r'\byes\b', text_lower):
        return "yes"
    if re.search(r'\bno\b', text_lower):
        return "no"
    return None

class ConsoleMetricsCallback(TrainerCallback):
    """
    Print selected metrics whenever Trainer logs.

    This keeps console output readable and helps track:
      - loss
      - learning rate schedule
      - eval mae/rmse if computed
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        keys_of_interest = ['loss', 'grad_norm', 'learning_rate', 'epoch', 'mae', 'rmse']
        selected = {k: logs[k] for k in keys_of_interest if k in logs}
        if selected:
            print(f"[METRICS] {json.dumps(selected)}")


class GradientMonitoringCallback(TrainerCallback):
    """
    Monitors gradient health during training by calling model.clip_and_monitor_gradients()
    every N steps.

    Requirements:
      PersonalLLM_Slim must implement:
        - clip_and_monitor_gradients(): returns dict with "grad_norm" (+ optional warning)
    """
    def __init__(self, check_every=50):
        super().__init__()
        self.check_every = check_every
        self.grad_norms = []
    
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "clip_and_monitor_gradients"):
            return
        
        if state.global_step % self.check_every != 0:
            return
        
        grad_monitor = model.clip_and_monitor_gradients()
        self.grad_norms.append(grad_monitor["grad_norm"])
        
        warning_msg = grad_monitor.get("warning")
        if warning_msg:
            print(f"\n⚠️ [Step {state.global_step}] {warning_msg}")
        else:
            grad_norm_val = grad_monitor["grad_norm"]
            print(f"✅ Step {state.global_step}: grad_norm={grad_norm_val:.6f}")


class GradientFlowCheckCallback(TrainerCallback):
    """
    Periodically prints a per-module gradient norm summary.

    This is a "deep" diagnostic: it helps confirm that the fusion stack is actually
    receiving gradients:
      - align MLPs (profile/session/graph)
      - gates
      - session encoder

    Note:
      This callback currently prints on EVERY backward_end call (no throttling).
      (It has check_every/min_grad_norm fields, but doesn't use them in the print.)
    """
    def __init__(self, check_every=50, min_grad_norm=1e-7):
        super().__init__()
        self.check_every = check_every
        self.min_grad_norm = min_grad_norm
    
    def on_backward_end(self, args, state, control, **kwargs):
        """Called after backward pass — check all modules have non-zero gradients."""
        model = kwargs.get('model')
        
        print("\n" + "="*60)
        print("🔍 GRADIENT FLOW VERIFICATION")
        print("="*60)
        
        modules_to_check = [
            'align_mlp', 'align_mlp_inst', 'align_mlp_session', 'align_mlp_graph',
            'gate_profile', 'gate_session', 'gate_graph',
            'session_encoder', 'session_pos_enc'
        ]
        
        for module_name in modules_to_check:
            module = getattr(model, module_name, None)
            if module is None:
                continue
            
            total_grad_norm = 0.0
            has_grad = False
            
            # Modules can be either Parameter or nn.Module
            if isinstance(module, torch.nn.Parameter):
                if module.grad is not None:
                    total_grad_norm = module.grad.norm().item()
                    has_grad = True
            else:
                for param in module.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        has_grad = True
                total_grad_norm = total_grad_norm ** 0.5
            
            status = "✅" if has_grad else "❌"
            print(f"{status} {module_name:30s} | grad_norm={total_grad_norm:.8f}")
        
        print("="*60 + "\n")


class TrainingDiagnosticsCallback(TrainerCallback):
    """
    Pulls higher-level training diagnostics from the model every 100 steps.

    Requirements:
      PersonalLLM_Slim must implement:
        - get_training_diagnostics(): returns dict with keys like
            loss_ema, grad_norm_mean, gate_mean
    """
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "get_training_diagnostics"):
            return
        
        if state.global_step % 100 == 0 and state.global_step > 0:
            diag = model.get_training_diagnostics()
            print(
                f"[Step {state.global_step}] "
                f"loss_ema={diag.get('loss_ema', 'N/A'):.4f} | "
                f"grad_norm={diag.get('grad_norm_mean', 'N/A'):.4f} | "
                f"gate_mean={diag.get('gate_mean', 'N/A'):.3f}"
            )


# -----------------------------
# Arguments Setup
# -----------------------------
@dataclass
class ModelArguments:
    """
    Model-side arguments (paths + ablation flags).

    The three ablation flags control WHICH personalization sources are used:
      - use_profile: long-term history embeddings (his_id)
      - use_session: short-term session transformer embeddings (session_ids)
      - use_graph:   graph embedding lookup (graph_node_ids)

    Quantization flags exist but are not used in this file (model may use them).
    """
    model_path: str = field(default="../flant5-small/", metadata={"help": "Path to the pretrain model."})
    emb_model_path: str = field(default="../bge-base-en-v1.5/", metadata={"help": "Path to the embedding model"})
    use_4bit: bool = field(default=False, metadata={"help": "Enable 4-bit BitsAndBytes quantization"})
    use_8bit: bool = field(default=False, metadata={"help": "Enable 8-bit BitsAndBytes quantization"})

    # ✅ A/B TESTING: Only these three are for source ablation
    # NOTE: HfArgumentParser bool parsing is strict; accept strings and convert via str2bool.
    use_profile: str = field(default="true", metadata={"help": "Use long-term profile/history embeddings (his_id)."})
    use_session: str = field(default="true", metadata={"help": "Use short-term session embeddings (session_ids)."})
    use_graph: str = field(default="true", metadata={"help": "Use graph embeddings (graph_node_ids)."})

@dataclass
class DataArguments:
    """
    Data-side arguments: dataset files and sequence lengths.

    task_id is optional; if not provided, we attempt to infer it from paths.
    """
    train_file: str = field(default="../data/train.json", metadata={"help": "Training dataset file path"})
    dev_file: str = field(default="../data/dev.json", metadata={"help": "Validation dataset file path"})
    max_input_len: int = field(default=256, metadata={"help": "Max input length"})
    max_new_len: int = field(default=32, metadata={"help": "Max new token length"})
    max_his_len: int = field(default=512, metadata={"help": "Max history length"})
    use_subset: bool = field(default=True, metadata={"help": "Use a smaller subset for quick testing (default=True)"})
    max_session_len: int = field(default=7, metadata={"help": "Maximum number of recent session items to use"})
    task_id: Optional[int] = field(default=None, metadata={"help": "LaMP task ID (1-7). Required for dataset loading."})


# -----------------------------
# Train Function
# -----------------------------
def train_model(model_args, data_args, training_args):

    # -------------------------------------------------------------------------
    # [1] Load tokenizers
    # -------------------------------------------------------------------------
    # LLM tokenizer (T5)
    llm_tokenizer = T5Tokenizer.from_pretrained(model_args.model_path)
    llm_tokenizer.model_max_length = data_args.max_input_len

    # Embedding tokenizer (BGE)
    emb_tokenizer = AutoTokenizer.from_pretrained(model_args.emb_model_path)
    emb_tokenizer.model_max_length = data_args.max_input_len

    # -------------------------------------------------------------------------
    # [2] Load base LLM (Flan-T5)
    # -------------------------------------------------------------------------
    llm_model_loaded = transformers.T5ForConditionalGeneration.from_pretrained(model_args.model_path)

    # Force config limits to match dataset cropping to avoid warning spam.
    # (T5 doesn't use absolute positions like BERT, but configs still exist.)
    llm_model_loaded.config.n_positions = data_args.max_input_len
    llm_model_loaded.config.max_position_embeddings = data_args.max_input_len
    llm_model_loaded.config.model_max_length = data_args.max_input_len

    def compute_metrics_classification(eval_preds):
        """
        Compute evaluation metrics from Seq2SeqTrainer outputs.

        HF returns (preds, labels) where preds may be:
          - token ids (batch, seq_len)
          - logits (batch, seq_len, vocab)
        We normalize this and decode to text, then extract:
          - rating in {1..5}
          - yes/no label

        Returns dict of metrics for Trainer logging (supports old keys mae/rmse/acc).
        """
        preds, labels = eval_preds
        
        # ---- Normalize preds array shape ----
        if isinstance(preds, np.ndarray):
            # If preds are logits: (B, T, V) -> argmax over V -> (B, T)
            if preds.ndim == 3:
                preds = np.argmax(preds, axis=-1)

            # If preds are token ids: (B, T) -> take first token
            # (This is a simplification for tasks where answer is a single token)
            if preds.ndim == 2:
                preds = preds[:, 0]
        
        # Ensure labels are 1D and aligned with our "first token" assumption
        if isinstance(labels, np.ndarray) and labels.ndim == 2:
            labels = labels[:, 0]
        
        # Remove padding token label -100 (HF convention)
        valid_mask = labels != -100
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        
        # Clamp negative ids defensively
        preds = np.maximum(preds, 0)

        # Decode token ids to strings
        pred_texts = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
        gold_texts = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Optional JSONL logging for per-sample debugging
        log_path = os.environ.get("PPLOG_EVAL_JSONL", "").strip()
        log_fp = open_jsonl(log_path) if log_path else None

        rating_preds, rating_golds = [], []
        yn_correct = 0
        yn_total = 0
        skipped = 0

        # Debug print (first 20)
        print("\n[DEBUG] Eval pred vs gold (first 20)")
        for i, (p, g) in enumerate(zip(pred_texts, gold_texts)):
            if i >= 20:
                break
            pr = extract_rating_1_to_5(p)
            gr = extract_rating_1_to_5(g)
            py = extract_yes_no(p)
            gy = extract_yes_no(g)
            print(f"Sample {i}: pred='{p}' (rating={pr}, yn={py}) | gold='{g}' (rating={gr}, yn={gy})")

        for i, (p, g) in enumerate(zip(pred_texts, gold_texts)):
            pr = extract_rating_1_to_5(p)
            gr = extract_rating_1_to_5(g)
            if pr is not None and gr is not None:
                rating_preds.append(float(pr))
                rating_golds.append(float(gr))

                if log_fp is not None:
                    log_eval_row(log_fp, {
                        "i": i,
                        "type": "rating",
                        "pred_text": str(p),
                        "gold_text": str(g),
                        "pred_rating": pr,
                        "gold_rating": gr,
                        "correct": bool(int(pr) == int(gr)),
                    })
                continue

            py = extract_yes_no(p)
            gy = extract_yes_no(g)
            if py is not None and gy is not None:
                yn_total += 1
                if py == gy:
                    yn_correct += 1

                if log_fp is not None:
                    log_eval_row(log_fp, {
                        "i": i,
                        "type": "yesno",
                        "pred_text": str(p),
                        "gold_text": str(g),
                        "pred_yesno": py,
                        "gold_yesno": gy,
                        "correct": bool(py == gy),
                    })
                continue

            skipped += 1
            if log_fp is not None:
                log_eval_row(log_fp, {
                    "i": i,
                    "type": "skipped",
                    "pred_text": str(p),
                    "gold_text": str(g),
                    "pred_rating": pr,
                    "gold_rating": gr,
                    "pred_yesno": py,
                    "gold_yesno": gy,
                })

        if log_fp is not None:
            log_fp.close()

        metrics = {"n_total": len(pred_texts), "n_skipped": skipped}

        # Rating metrics (regression)
        if rating_preds:
            n = len(rating_preds)
            mae = sum(abs(p - r) for p, r in zip(rating_preds, rating_golds)) / n
            rmse = math.sqrt(sum((p - r) ** 2 for p, r in zip(rating_preds, rating_golds)) / n)
            acc = sum(1 for p, r in zip(rating_preds, rating_golds) if int(p) == int(r)) / n
            mape = sum(abs((p - r) / r) if r != 0 else 0.0 for p, r in zip(rating_preds, rating_golds)) / n
            metrics.update({
                "rating_acc": acc,
                "rating_mae": mae,
                "rating_rmse": rmse,
                "rating_mape": mape,
                "n_rating": n
            })
        else:
            metrics.update({"n_rating": 0})

        # Yes/No metrics (classification)
        if yn_total:
            metrics.update({
                "yn_acc": yn_correct / yn_total,
                "n_yn": yn_total
            })
        else:
            metrics.update({"n_yn": 0})

        # Backwards-compat keys so Trainer logs don't break existing plots:
        # If we have rating metrics, expose them as mae/rmse. Otherwise fall back to yn_acc.
        if metrics["n_rating"] > 0:
            metrics["mae"] = metrics["rating_mae"]
            metrics["rmse"] = metrics["rating_rmse"]
            metrics["mape"] = metrics["rating_mape"]
            metrics["acc"] = metrics["rating_acc"]
        elif metrics["n_yn"] > 0:
            metrics["acc"] = metrics["yn_acc"]

        return metrics

    emb_model = AutoModel.from_pretrained(model_args.emb_model_path)

    # -------------------------------------------------------------------------
    # [5] Determine task_id (needed to locate graph artifacts)
    # -------------------------------------------------------------------------
    # Priority:
    #  - user-provided --task_id
    #  - parse from file paths "...LaMP_time_X..."
    #  - parse from output dir
    if data_args.task_id is not None:
        task_id = data_args.task_id
    else:
        match = re.search(r'LaMP_time_(\d+)', data_args.train_file)
        if match:
            task_id = int(match.group(1))
        else:
            match = re.search(r'LaMP_time_(\d+)', data_args.dev_file)
            if match:
                task_id = int(match.group(1))
            else:
                match = re.search(r'task(\d+)|_(\d+)_', training_args.output_dir)
                if match:
                    task_id = int(match.group(1) or match.group(2))
                else:
                    raise ValueError(
                        "Cannot determine task_id. Please provide --task_id argument "
                        "or ensure train_file/dev_file contain 'LaMP_time_X' pattern."
                    )
    
    print(f"📋 Task ID: {task_id}")

    # -------------------------------------------------------------------------
    # [6] Graph artifacts (precomputed offline)
    # -------------------------------------------------------------------------
    # The extension pipeline precomputes:
    #   - task_{id}_graph.npy                  (node embeddings)
    #   - task_{id}_his_to_graph(_node).json   (his_id -> graph node id mapping)
    GRAPH_DIR = "../graph_emb"

    graph_emb_npy_path = os.path.join(GRAPH_DIR, f"task_{task_id}_graph.npy")

    mapping_candidates = [
        os.path.join(GRAPH_DIR, f"task_{task_id}_his_to_graph_node.json"),
        os.path.join(GRAPH_DIR, f"task_{task_id}_his_to_graph.json"),
    ]
    his_to_graph_path = next((p for p in mapping_candidates if os.path.exists(p)), None)

    if his_to_graph_path is None:
        raise FileNotFoundError(
            f"Missing his_to_graph mapping. Tried: {mapping_candidates}"
        )

    train_dataset = PersonalDataset(
        data_args.train_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer,
        graph_emb_path=graph_emb_npy_path,
        his_to_graph_path=his_to_graph_path,
        max_session_len=getattr(data_args, "max_session_len", 3),
    )
    eval_dataset = PersonalDataset(
        data_args.dev_file, data_args.max_input_len, data_args.max_new_len,
        data_args.max_his_len, llm_tokenizer, emb_tokenizer,
        graph_emb_path=graph_emb_npy_path,
        his_to_graph_path=his_to_graph_path,
        max_session_len=getattr(data_args, "max_session_len", 3),
    )

    # -------------------------------------------------------------------------
    # [8] Initialize the wrapper model (PersonalLLM_Slim)
    # -------------------------------------------------------------------------
    # A/B testing flags control which personalization sources contribute.
    # "Internal architecture" flags are set to ON (always use fusion components).
    #PersonalLLM_Slim_StageAB or PersonalLLM_Slim
    model = PersonalLLM_Slim_StageAB(
        llm_model=llm_model_loaded,
        emb_model=emb_model,
        llm_tokenizer=llm_tokenizer,
        max_input_len=data_args.max_input_len,
        max_new_len=data_args.max_new_len,
        task_id=task_id,

        # A/B testing: which sources are active
        use_profile=model_args.use_profile,
        use_session=model_args.use_session,
        use_graph=model_args.use_graph,

        # Fusion stack toggles (kept always ON in this training script)
        use_inst_token=True,
        use_align_mlp_inst=True,
        use_align_mlp=True,
        use_align_mlp_session=True,
        use_align_mlp_graph=True,
        use_session_encoder=True,
        use_cross_attn=True,
        use_gate=True,
    )

    # Important: tokenizer has custom special tokens ([INST_PER_TOKEN], [SPC_PER_TOKEN]).
    # Need to resize embeddings so T5 can accept them.
    model.llm_model.resize_token_embeddings(len(llm_tokenizer))

    # -------------------------------------------------------------------------
    # [9] Configure stability knobs on the wrapper model
    # -------------------------------------------------------------------------
    # These are custom attributes used by PersonalLLM_Slim (not standard HF).
    model.max_grad_norm = 1.0
    model.grad_norm_check_steps = 50
    model.warmup_steps = min(1000, len(train_dataset) // training_args.per_device_train_batch_size)
    model.loss_ema_alpha = 0.05

    # Also configure Trainer args for stability:
    training_args.max_grad_norm = 1.0
    training_args.gradient_checkpointing = False
    training_args.fp16 = False
    training_args.bf16 = False
    training_args.logging_first_step = True
    training_args.warmup_ratio = 0.1
    training_args.weight_decay = 1e-4
    training_args.learning_rate = 3e-5

    # -------------------------------------------------------------------------
    # [10] Ensure gate params are trainable (defensive)
    # -------------------------------------------------------------------------
    if hasattr(model, 'gate') and model.gate is not None:
        for param in model.gate.parameters():
            param.requires_grad = True
        print("✅ Gate is trainable")
    
    if hasattr(model, 'gate_temperature') and model.gate_temperature is not None:
        model.gate_temperature.requires_grad = True
        print("✅ Gate temperature is trainable")

    # ✅ Print which fusion sources are active
    print("\n" + "="*60)
    print("📊 ACTIVE FUSION SOURCES")
    print("="*60)
    print(f"  use_profile: {model.use_profile}")
    print(f"  use_session: {model.use_session}")
    print(f"  use_graph: {model.use_graph}")
    print(f"  use_inst_token: {model.use_inst_token}")
    print(f"  use_gate: {model.use_gate}")
    print("="*60 + "\n")

    # ✅ Print parameter summary
    print("\n" + "="*60)
    print("📊 MODEL PARAMETER SUMMARY")
    print("="*60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable ratio: {100*trainable_params/total_params:.2f}%")
    print("="*60 + "\n")

    # -------------------------------------------------------------------------
    # [11] Patch: Seq2SeqTrainer expects model.generate()
    # -------------------------------------------------------------------------
    # Our wrapper model may not implement generate; the underlying T5 does.
    # Expose a proxy so Trainer can call generate() during eval/predict.
    if not hasattr(model, "generate") and hasattr(model, "llm_model") and hasattr(model.llm_model, "generate"):
        def generate(self, *args, **kwargs):
            return self.llm_model.generate(*args, **kwargs)
        model.generate = generate.__get__(model, model.__class__)

    # -------------------------------------------------------------------------
    # [12] Data collator: pad variable-length tensors in a batch
    # -------------------------------------------------------------------------
    # Note: the dataset already emits fixed sizes for many fields,
    # but labels/inputs can still differ depending on truncation settings,
    # so we pad to longest in batch for safety.
    def my_collator(features):
        def pad_to_longest(key, dtype=torch.long, pad_value=0):
            # Handle missing keys gracefully (e.g., dataset doesn't emit graph fields)
            if not any(key in f for f in features):
                return None
            
            tensors = [
                torch.as_tensor(f.get(key, []), dtype=dtype) 
                for f in features
            ]
            return pad_sequence(
                tensors,
                batch_first=True,
                padding_value=pad_value
            )
        
        batch = {
            'llm_input_ids': pad_to_longest('llm_input_ids'),
            'llm_attention_mask': pad_to_longest('llm_attention_mask'),
            'labels': pad_to_longest('labels'),
            'emb_input_ids': pad_to_longest('emb_input_ids'),
            'emb_attention_mask': pad_to_longest('emb_attention_mask'),
            'emb_token_type_ids': pad_to_longest('emb_token_type_ids'),
            'his_id': pad_to_longest('his_id'),
            'session_ids': pad_to_longest('session_ids'),

            # Graph padding conventions:
            #   - ids: pad with -1 (meaning "invalid node")
            #   - mask: pad with 0 (since dtype=torch.long and pad_value default is 0)
            'graph_node_ids': pad_to_longest('graph_node_ids', pad_value=-1),
            'graph_node_mask': pad_to_longest('graph_node_mask', dtype=torch.long)
        }
        
        # Drop missing keys (None) to avoid passing them into the model
        batch = {k: v for k, v in batch.items() if v is not None}
        
        return batch

    # -------------------------------------------------------------------------
    # [13] Build rating-only logits processors for constrained generation
    # -------------------------------------------------------------------------
    allowed_rating_token_ids = _get_allowed_rating_token_ids(llm_tokenizer)
    rating_logits_processors = RatingOnlyLogitsProcessorList()
    rating_logits_processors.append(_RestrictToRatingTokensProcessor(allowed_rating_token_ids))

    # -------------------------------------------------------------------------
    # [14] Pre-training "smoke test": forward + backward on 1 batch
    # -------------------------------------------------------------------------
    # Goal: fail fast if shapes/ids are wrong (common with graph/session ids).
    print("\n" + "="*60)
    print("🔍 PRE-TRAINING GRADIENT FLOW TEST")
    print("="*60)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=my_collator
    )

    test_batch = next(iter(train_loader))

    # Move tensors to device (cpu/cuda)
    for key in test_batch:
        if isinstance(test_batch[key], torch.Tensor):
            test_batch[key] = test_batch[key].to(training_args.device)

    # Filter out None values defensively
    test_batch = {k: v for k, v in test_batch.items() if v is not None}

    # Forward pass
    print(f"\n📈 Forward pass:")
    model.zero_grad()
    
    try:
        outputs = model(**test_batch)
    except IndexError as e:
        # IndexError is common when embedding table indices go out of range.
        # This block prints input shapes to help identify which tensor is bad.
        print(f"❌ IndexError during forward pass: {e}")
        print(f"\nDebug info:")
        for k, v in test_batch.items():
            shape_info = f"shape={v.shape}" if hasattr(v, 'shape') else f"type={type(v)}"
            print(f"  {k}: {shape_info}")
        raise
    
    loss = outputs["loss"]

    if loss is None or loss.grad_fn is None:
        print("❌ CRITICAL: Loss has no grad_fn! Forward pass broken.")
        print(f"   Loss: {loss}")
        print(f"   Loss type: {type(loss)}")
        exit(1)

    print(f"   ✅ Loss: {loss.item():.6f} | requires_grad: {loss.requires_grad}")

    # Backward pass
    print(f"\n📉 Backward pass:")
    loss.backward()

    # Check gradient flow
    grad_info = model.clip_and_monitor_gradients()
    print(f"   ✅ Gradient norm: {grad_info['grad_norm']:.8f}")

    if grad_info['grad_norm'] < 1e-8:
        print(f"   ❌ WARNING: Gradients extremely small!")
    elif grad_info['grad_norm'] > 10.0:
        print(f"   ⚠️ WARNING: Gradients very large (will be clipped)")
    else:
        print(f"   ✅ Gradients look healthy!")

    # Check specific fusion layers
    print(f"\n🔍 Fusion layer gradients:")
    fusion_modules = ['align_mlp', 'align_mlp_inst', 'align_mlp_session', 'align_mlp_graph', 'gate']
    
    for module_name in fusion_modules:
        module = getattr(model, module_name, None)
        if module is None:
            print(f"   ⚠️  {module_name}: NOT FOUND")
            continue

        total_norm = 0.0
        param_count = 0
        has_grad = False

        for param in module.parameters():
            if param.requires_grad and param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
                param_count += 1
                has_grad = True

        total_norm = (total_norm ** 0.5) if param_count > 0 else 0.0
        status = "✅" if has_grad else "❌"
        print(f"   {status} {module_name:25s} | grad_norm={total_norm:.8f}")

    model.zero_grad()
    print("="*60 + "\n")

    # ============================================================
    # Initialize Trainer with Callbacks
    # ============================================================
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=llm_tokenizer,
        data_collator=my_collator,
        compute_metrics=compute_metrics_classification,
        callbacks=[
            ConsoleMetricsCallback(),
            FusionDataDiagnosticsCallback(),
            GradientMonitoringCallback(check_every=50),  # ✅ Watch gradients
            GradientFlowCheckCallback(check_every=50),
            TrainingDiagnosticsCallback(),
        ]
    )

    # -------------------------------------------------------------------------
    # [16] Train
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    print(f"Max gradient norm: {training_args.max_grad_norm}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Warmup ratio: {training_args.warmup_ratio}")
    print("="*60 + "\n")

    trainer.train()

    # -------------------------------------------------------------------------
    # [17] Final diagnostics (gate behavior / gradient summary)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("📊 FINAL TRAINING REPORT")
    print("="*60)
    
    final_diag = model.get_training_diagnostics()
    print(f"Loss EMA: {final_diag.get('loss_ema', 'N/A'):.6f}")
    print(f"Avg Gradient Norm: {final_diag.get('grad_norm_mean', 'N/A'):.6f}")
    print(f"Gate Mean: {final_diag.get('gate_mean', 'N/A'):.4f} (target: 0.3-0.7)")
    
    # Gate mean too high/low suggests the model collapsed to always/never personalize.
    if final_diag.get('gate_mean', 1.0) > 0.95:
        print("⚠️ WARNING: Gate stuck at high value (all personalization)")
    elif final_diag.get('gate_mean', 0.0) < 0.05:
        print("⚠️ WARNING: Gate stuck at low value (ignoring personalization)")
    else:
        print("✅ Gate is blending task and personalization properly!")
    
    print("="*60 + "\n")

    # -------------------------------------------------------------------------
    # [18] Patch: enforce rating-only decoding during evaluation
    # -------------------------------------------------------------------------
    # This overrides trainer.model.generate to always apply our logits_processor.
    try:
        trainer.model.generation_config.max_length = 2
        trainer.model.generation_config.num_beams = 5
        trainer.model.generation_config.do_sample = False
    except Exception:
        pass

    _orig_generate = trainer.model.generate

    def _rating_only_generate(*args, **kwargs):
        kwargs.setdefault("max_length", 2)
        kwargs.setdefault("num_beams", 5)
        kwargs.setdefault("do_sample", False)
        kwargs["logits_processor"] = rating_logits_processors
        return _orig_generate(*args, **kwargs)

    trainer.model.generate = _rating_only_generate

    # -------------------------------------------------------------------------
    # [19] Evaluate
    # -------------------------------------------------------------------------
    print("🔍 Running final evaluation...")
    outputs = trainer.evaluate()
    print(outputs)
    
    # Save training diagnostics JSON for later plotting/debugging
    diagnostics_path = os.path.join(training_args.output_dir, "training_diagnostics.json")
    with open(diagnostics_path, 'w') as f:
        json.dump(model.get_training_diagnostics(), f, indent=2)
    print(f"✅ Training diagnostics saved to {diagnostics_path}")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    transformers.set_seed(42)

    # Parse three groups:
    #  - ModelArguments
    #  - DataArguments
    #  - Seq2SeqTrainingArguments (HF Trainer built-ins)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Convert string flags -> real booleans (robust to True/False/1/0/yes/no)
    model_args.use_profile = str2bool(model_args.use_profile)
    model_args.use_session = str2bool(model_args.use_session)
    model_args.use_graph = str2bool(model_args.use_graph)

    print(f"[ARGS] use_profile={model_args.use_profile} use_session={model_args.use_session} use_graph={model_args.use_graph}")

    # This repo uses classic .bin checkpoints; disable safetensors to match.
    training_args.save_safetensors = False

    train_model(model_args, data_args, training_args)