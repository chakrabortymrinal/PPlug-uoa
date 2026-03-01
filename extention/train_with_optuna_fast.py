
"""
Optuna Hyperparameter Optimization for PersonalLLM_Slim_StageAB

Strategy for 40-min training runs:
  - Use MedianPruner to kill bad trials after Epoch 1-2 (~13-26 mins)
  - Sample reduced hyperparameter space to finish in 24-48 hours
  - Report intermediate MAE after each epoch for early stopping
  - Save best trial checkpoint for production use

Expected runtime:
  - Good trials: 40 mins (3 epochs)
  - Bad trials: 13-26 mins (pruned after 1-2 epochs)
  - ~30 trials in 24 hours with 50% pruning rate
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    AutoModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

# Local imports (adjust paths if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extention.PersonalDataset_profile_GNN import PersonalDataset
from extention.ModelForPer_slim_GNN_stageA_B import (
    PersonalLLM_Slim_StageAB,
    FinetuneConfig,
    TrainHyperparams,
)
from extention.eval_logging import extract_rating_1_to_5


# =============================================================================
# Helper: Safe Device Detection
# =============================================================================
def get_safe_device():
    """
    Get training device with MPS fallback handling.
    
    MPS (Apple Silicon GPU) has known issues with:
    - Certain indexing operations
    - Placeholder storage allocation
    - Mixed precision training
    
    Returns CPU for stability on Mac.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("⚠️  MPS device detected but using CPU for stability")
        print("   (MPS has known issues with graph indexing operations)")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


# =============================================================================
# Custom Data Collator for StageAB Model
# =============================================================================
@dataclass
class StageABDataCollator:
    """
    Custom collator that handles multi-modal inputs:
      - llm_input_ids, llm_attention_mask, labels
      - emb_input_ids, emb_attention_mask, emb_token_type_ids
      - his_id, session_ids, graph_node_ids, graph_node_mask
    
    Handles both list and tensor inputs from dataset.
    """
    
    llm_tokenizer: Any
    padding: str = "longest"
    max_length: int = None
    pad_to_multiple_of: int = None
    label_pad_token_id: int = -100
    
    @staticmethod
    def _to_list(x):
        """Convert tensor or list to Python list"""
        if isinstance(x, torch.Tensor):
            return x.tolist()
        return list(x)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples with custom keys"""
        
        batch = {}
        
        # -------------------------------------------------------------------------
        # 1️⃣ LLM inputs (input_ids, attention_mask) - requires padding
        # -------------------------------------------------------------------------
        llm_input_ids = [self._to_list(f["llm_input_ids"]) for f in features]
        llm_attention_mask = [self._to_list(f["llm_attention_mask"]) for f in features]
        
        # Pad to longest in batch
        max_len = max(len(ids) for ids in llm_input_ids)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_llm_input_ids = []
        padded_llm_attention_mask = []
        
        for ids, mask in zip(llm_input_ids, llm_attention_mask):
            padding_len = max_len - len(ids)
            padded_llm_input_ids.append(
                ids + [self.llm_tokenizer.pad_token_id] * padding_len
            )
            padded_llm_attention_mask.append(
                mask + [0] * padding_len
            )
        
        # ✅ FIX: Use custom key names that match model.forward() signature
        batch["llm_input_ids"] = torch.tensor(padded_llm_input_ids, dtype=torch.long)
        batch["llm_attention_mask"] = torch.tensor(padded_llm_attention_mask, dtype=torch.long)
        
        # -------------------------------------------------------------------------
        # 2️⃣ Labels (target outputs) - pad with -100
        # -------------------------------------------------------------------------
        if "labels" in features[0]:
            labels = [self._to_list(f["labels"]) for f in features]
            max_label_len = max(len(l) for l in labels)
            
            padded_labels = []
            for l in labels:
                padding_len = max_label_len - len(l)
                padded_labels.append(l + [self.label_pad_token_id] * padding_len)
            
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        # -------------------------------------------------------------------------
        # 3️⃣ Embedding inputs (same padding logic)
        # -------------------------------------------------------------------------
        emb_input_ids = [self._to_list(f["emb_input_ids"]) for f in features]
        emb_attention_mask = [self._to_list(f["emb_attention_mask"]) for f in features]
        emb_token_type_ids = [
            self._to_list(f.get("emb_token_type_ids", [0]*len(f["emb_input_ids"]))) 
            for f in features
        ]
        
        max_emb_len = max(len(ids) for ids in emb_input_ids)
        
        padded_emb_input_ids = []
        padded_emb_attention_mask = []
        padded_emb_token_type_ids = []
        
        for ids, mask, ttids in zip(emb_input_ids, emb_attention_mask, emb_token_type_ids):
            padding_len = max_emb_len - len(ids)
            padded_emb_input_ids.append(ids + [0] * padding_len)
            padded_emb_attention_mask.append(mask + [0] * padding_len)
            padded_emb_token_type_ids.append(ttids + [0] * padding_len)
        
        batch["emb_input_ids"] = torch.tensor(padded_emb_input_ids, dtype=torch.long)
        batch["emb_attention_mask"] = torch.tensor(padded_emb_attention_mask, dtype=torch.long)
        batch["emb_token_type_ids"] = torch.tensor(padded_emb_token_type_ids, dtype=torch.long)
        
        # -------------------------------------------------------------------------
        # 4️⃣ Profile history IDs (already fixed-length from dataset)
        # -------------------------------------------------------------------------
        if "his_id" in features[0]:
            his_ids = [self._to_list(f["his_id"]) for f in features]
            batch["his_id"] = torch.tensor(his_ids, dtype=torch.long)
        
        # -------------------------------------------------------------------------
        # 5️⃣ Session IDs (optional, fixed-length)
        # -------------------------------------------------------------------------
        if "session_ids" in features[0]:
            session_ids = [self._to_list(f["session_ids"]) for f in features]
            batch["session_ids"] = torch.tensor(session_ids, dtype=torch.long)
        
        # -------------------------------------------------------------------------
        # 6️⃣ Graph inputs (optional, fixed-length) - ✅ KEEP ON CPU
        # -------------------------------------------------------------------------
        if "graph_node_ids" in features[0]:
            graph_node_ids = [self._to_list(f["graph_node_ids"]) for f in features]
            # ✅ CRITICAL: Keep on CPU to match graph_node_emb device
            batch["graph_node_ids"] = torch.tensor(graph_node_ids, dtype=torch.long, device="cpu")
        
        if "graph_node_mask" in features[0]:
            graph_node_mask = [self._to_list(f["graph_node_mask"]) for f in features]
            # ✅ CRITICAL: Keep on CPU
            batch["graph_node_mask"] = torch.tensor(graph_node_mask, dtype=torch.float32, device="cpu")
        
        return batch


# =============================================================================
# Optuna Callback: Report metrics after each epoch
# =============================================================================
class OptunaReportCallback(TrainerCallback):
    """Report eval MAE to Optuna and prune if needed"""
    
    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self.epoch_count = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after each evaluation"""
        if metrics is None:
            return
            
        self.epoch_count += 1
        mae = metrics.get("eval_rating_mae", float('inf'))
        
        # ✅ Report intermediate value to Optuna
        self.trial.report(mae, step=self.epoch_count)
        
        # ✅ Check if trial should be pruned
        if self.trial.should_prune():
            print(f"\n🔪 Trial {self.trial.number} PRUNED at epoch {self.epoch_count} (MAE={mae:.3f})")
            raise optuna.TrialPruned()





# =============================================================================
# Custom Trainer: CPU-Safe for Graph Tensors
# =============================================================================
class CPUSafeSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom trainer that keeps graph tensors on CPU even when model is on GPU.
    
    Overrides _prepare_inputs to selectively move tensors:
    - Most tensors → model device (GPU/CPU)
    - graph_node_ids, graph_node_mask → always CPU (for CPU-resident graph_node_emb)
    """
    
    def _prepare_inputs(self, inputs):
        """
        Move inputs to device, BUT keep graph tensors on CPU.
        """
        # Get device where most of the model lives
        model_device = next(self.model.parameters()).device
        
        prepared = {}
        
        for key, value in inputs.items():
            if value is None:
                prepared[key] = None
                continue
            
            # ✅ CRITICAL: Keep graph inputs on CPU (they index CPU-resident embeddings)
            if key in ["graph_node_ids", "graph_node_mask"]:
                if isinstance(value, torch.Tensor):
                    prepared[key] = value.cpu()  # Force CPU
                else:
                    prepared[key] = value
            else:
                # Normal behavior: move to model device
                if isinstance(value, torch.Tensor):
                    prepared[key] = value.to(model_device)
                else:
                    prepared[key] = value
        
        return prepared


# =============================================================================
# Optuna Objective Function
# =============================================================================
def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Single trial: train StageAB model with sampled hyperparameters.
    
    Returns:
      Final eval MAE (lower is better)
    """
    
    # ✅ FORCE CPU-ONLY TRAINING (disable MPS completely)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Monkey-patch to disable MPS detection
    import torch.backends.mps
    _original_is_available = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False
    
    print(f"\n{'='*70}")
    print(f"🔬 TRIAL {trial.number} START")
    print(f"⚠️  MPS disabled - using CPU for stability")
    print(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # 1️⃣ Sample hyperparameters (REDUCED SEARCH SPACE)
    # -------------------------------------------------------------------------
    
    # Learning rates (log scale for better exploration)
    lr_gate = trial.suggest_float("lr_gate", 1e-4, 1e-3, log=True)
    lr_cross_attn = trial.suggest_float("lr_cross_attn", 5e-5, 5e-4, log=True)
    lr_align = trial.suggest_float("lr_align", 5e-5, 5e-4, log=True)
    lr_session = trial.suggest_float("lr_session", 5e-5, 5e-4, log=True)
    
    # Gate initialization (critical for stability)
    gate_bias_init = trial.suggest_float("gate_bias_init", -1.0, 0.5)
    gate_temp_init = trial.suggest_float("gate_temp_init", 0.3, 1.5)
    
    # Graph gate hyperparameters (NEW)
    graph_gate_bias_init = trial.suggest_float("graph_gate_bias_init", -1.0, 1.0)
    graph_gate_temp_init = trial.suggest_float("graph_gate_temp_init", 0.5, 2.0)
    graph_gate_boost_weight = trial.suggest_float("graph_gate_boost_weight", 0.1, 0.5)
    
    # Training dynamics
    warmup_steps = trial.suggest_int("warmup_steps", 50, 300, step=50)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    
    # Session encoder architecture
    session_num_layers = trial.suggest_int("session_num_layers", 1, 3)
    session_num_heads = trial.suggest_categorical("session_num_heads", [2, 4, 8])
    
    print(f"\n📊 Sampled Hyperparameters:")
    print(f"  lr_gate          : {lr_gate:.2e}")
    print(f"  lr_cross_attn    : {lr_cross_attn:.2e}")
    print(f"  lr_align         : {lr_align:.2e}")
    print(f"  lr_session       : {lr_session:.2e}")
    print(f"  gate_bias_init   : {gate_bias_init:.3f}")
    print(f"  gate_temp_init   : {gate_temp_init:.3f}")
    print(f"  graph_gate_bias  : {graph_gate_bias_init:.3f}")
    print(f"  graph_gate_temp  : {graph_gate_temp_init:.3f}")
    print(f"  graph_gate_boost : {graph_gate_boost_weight:.3f}")
    print(f"  warmup_steps     : {warmup_steps}")
    print(f"  max_grad_norm    : {max_grad_norm:.2f}")
    print(f"  weight_decay     : {weight_decay:.3f}")
    print(f"  session_layers   : {session_num_layers}")
    print(f"  session_heads    : {session_num_heads}")
    
    # -------------------------------------------------------------------------
    # 2️⃣ Load tokenizers and base models (shared across trials)
    # -------------------------------------------------------------------------
    
    llm_model_path = args.llm_model_path
    emb_model_path = args.emb_model_path
    
    llm_tokenizer = T5Tokenizer.from_pretrained(llm_model_path, legacy=False)
    emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
    
    # Load fresh base models for this trial (avoid parameter pollution)
    from transformers import T5ForConditionalGeneration
    llm_base = T5ForConditionalGeneration.from_pretrained(llm_model_path)
    emb_base = AutoModel.from_pretrained(emb_model_path)
    
    # ✅ Use safe device detection (handles MPS issues)
    device = get_safe_device()
    print(f"\n🖥️  Using device: {device}")
    
    # -------------------------------------------------------------------------
    # 3️⃣ Create StageAB model with trial hyperparameters
    # -------------------------------------------------------------------------
    
    train_hp = TrainHyperparams(
        lr_gate=lr_gate,
        lr_cross_attn=lr_cross_attn,
        lr_align=lr_align,
        lr_session=lr_session,
        gate_bias_init=gate_bias_init,
        gate_temperature_init=gate_temp_init,
        graph_gate_bias_init=graph_gate_bias_init,
        graph_gate_temperature_init=graph_gate_temp_init,
        graph_gate_boost_weight=graph_gate_boost_weight,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
    )
    
    finetune_cfg = FinetuneConfig(
        tune_llm=False,
        tune_emb=False,
        tune_inst_token=True,
        tune_align_mlps=True,
        tune_session_encoder=True,
        tune_cross_attn=True,
        tune_gate=True,
    )
    
    model = PersonalLLM_Slim_StageAB(
        llm_model=llm_base,
        emb_model=emb_base,
        llm_tokenizer=llm_tokenizer,
        max_input_len=args.max_input_len,
        max_new_len=args.max_new_len,
        task_id=args.task_id,
        use_profile=args.use_profile,
        use_session=args.use_session,
        use_graph=args.use_graph,
        session_num_layers=session_num_layers,
        session_num_heads=session_num_heads,
        graph_dir=args.graph_dir,
        bge_emb_dir=args.bge_emb_dir,
        finetune=finetune_cfg,
        train_hp=train_hp,
    )
    
    # ✅ Move model to device (graph_node_emb stays on CPU automatically)
    model = model.to(device)
    print(f"✅ Model initialized and moved to {device}")
    
    # ✅ CRITICAL: Force graph processing modules back to CPU
    if hasattr(model, 'align_mlp_graph') and model.align_mlp_graph is not None:
        model.align_mlp_graph.to('cpu')
        print(f"  ⚠️  align_mlp_graph forced to CPU (MPS incompatible)")
    
    if hasattr(model, 'graph_encoder') and model.graph_encoder is not None:
        model.graph_encoder.to('cpu')
        print(f"  ⚠️  graph_encoder forced to CPU (MPS incompatible)")
    
    # -------------------------------------------------------------------------
    # 4️⃣ Load datasets
    # -------------------------------------------------------------------------
    
    train_dataset = PersonalDataset(
        args.train_file,
        args.max_input_len,
        args.max_new_len,
        args.max_his_len,
        llm_tokenizer,
        emb_tokenizer,
        graph_emb_path=os.path.join(args.graph_dir, f"task_{args.task_id}_graph.npy"),
        his_to_graph_path=os.path.join(args.graph_dir, f"task_{args.task_id}_his_to_graph_node.json"),
        max_session_len=args.max_session_len,
    )
    
    eval_dataset = PersonalDataset(
        args.eval_file,
        args.max_input_len,
        args.max_new_len,
        args.max_his_len,
        llm_tokenizer,
        emb_tokenizer,
        graph_emb_path=os.path.join(args.graph_dir, f"task_{args.task_id}_graph.npy"),
        his_to_graph_path=os.path.join(args.graph_dir, f"task_{args.task_id}_his_to_graph_node.json"),
        max_session_len=args.max_session_len,
    )
    
    # Optional subsampling
    if args.subsample_train > 0:
        indices = np.random.choice(len(train_dataset), min(args.subsample_train, len(train_dataset)), replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    if args.subsample_eval > 0:
        indices = np.random.choice(len(eval_dataset), min(args.subsample_eval, len(eval_dataset)), replace=False)
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
    
    # -------------------------------------------------------------------------
    # 5️⃣ Training setup
    # -------------------------------------------------------------------------
    
    output_dir = os.path.join(args.optuna_output_dir, f"trial_{trial.number}")
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        
        eval_strategy="epoch",
        save_strategy="no",
        
        logging_steps=args.logging_steps,
        logging_dir=None,
        report_to="none",
        
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        
        fp16=False,  # ✅ Disable FP16 for CPU
        dataloader_num_workers=0,
        load_best_model_at_end=False,
        metric_for_best_model="rating_mae",
        greater_is_better=False,
        
        push_to_hub=False,
        save_total_limit=1,
    )
    
    # -------------------------------------------------------------------------
    # 6️⃣ Create optimizer
    # -------------------------------------------------------------------------
    
    param_groups = model.get_param_groups()
    optimizer = torch.optim.AdamW(param_groups)
    
    # -------------------------------------------------------------------------
    # 7️⃣ Compute metrics function
    # -------------------------------------------------------------------------
    
    def compute_metrics(eval_preds):
        """Compute rating metrics (MAE, RMSE, Accuracy)"""
        import math
        from extention.eval_logging import extract_rating_1_to_5
        
        preds, labels = eval_preds
        
        # Normalize predictions (handle logits vs token ids)
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # If 3D (logits), take argmax
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        
        # Take first token (single-token rating prediction)
        if preds.ndim == 2:
            preds = preds[:, 0]
        
        # Same for labels
        if isinstance(labels, np.ndarray) and labels.ndim == 2:
            labels = labels[:, 0]
        
        # Filter out padding (-100)
        valid_mask = labels != -100
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        
        # Decode to text
        pred_texts = llm_tokenizer.batch_decode(preds, skip_special_tokens=True)
        gold_texts = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Extract numeric ratings
        rating_preds = []
        rating_golds = []
        
        for p_text, g_text in zip(pred_texts, gold_texts):
            p_rating = extract_rating_1_to_5(p_text)
            g_rating = extract_rating_1_to_5(g_text)
            
            if p_rating is not None and g_rating is not None:
                rating_preds.append(float(p_rating))
                rating_golds.append(float(g_rating))
        
        # Compute metrics
        if len(rating_preds) == 0:
            return {
                "rating_mae": 999.0,
                "rating_rmse": 999.0,
                "rating_accuracy": 0.0,
                "n_valid": 0,
            }
        
        n = len(rating_preds)
        mae = sum(abs(p - g) for p, g in zip(rating_preds, rating_golds)) / n
        rmse = math.sqrt(sum((p - g) ** 2 for p, g in zip(rating_preds, rating_golds)) / n)
        accuracy = sum(1 for p, g in zip(rating_preds, rating_golds) if int(p) == int(g)) / n
        
        return {
            "rating_mae": mae,
            "rating_rmse": rmse,
            "rating_accuracy": accuracy,
            "n_valid": n,
        }
    
    # -------------------------------------------------------------------------
    # 8️⃣ Trainer with CPU-safe graph handling
    # -------------------------------------------------------------------------
    
    data_collator = StageABDataCollator(
        llm_tokenizer=llm_tokenizer,
        padding="longest",
        pad_to_multiple_of=None,  # ✅ Disable for CPU
        label_pad_token_id=-100,
    )
    
    # ✅ Use custom trainer that keeps graph tensors on CPU
    trainer = CPUSafeSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=llm_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[OptunaReportCallback(trial)],
    )
    
    # -------------------------------------------------------------------------
    # 9️⃣ Train (will raise TrialPruned if bad)
    # -------------------------------------------------------------------------
    
    try:
        print(f"\n🚀 Training trial {trial.number}...")
        trainer.train()
        
    except optuna.TrialPruned:
        print(f"\n✂️  Trial {trial.number} was pruned early")
        raise
    
    # -------------------------------------------------------------------------
    # 🔟 Final evaluation
    # -------------------------------------------------------------------------
    
    print(f"\n📊 Final evaluation for trial {trial.number}...")
    metrics = trainer.evaluate()
    
    final_mae = metrics["eval_rating_mae"]
    final_rmse = metrics["eval_rating_rmse"]
    final_acc = metrics["eval_rating_accuracy"]
    
    print(f"\n✅ Trial {trial.number} COMPLETE:")
    print(f"  Final MAE  : {final_mae:.4f}")
    print(f"  Final RMSE : {final_rmse:.4f}")
    print(f"  Final Acc  : {final_acc:.4f}")
    
    # Save trial summary
    trial_summary = {
        "trial_number": trial.number,
        "hyperparameters": trial.params,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "final_accuracy": final_acc,
        "num_epochs": args.num_epochs,
    }
    
    with open(os.path.join(output_dir, "trial_summary.json"), "w") as f:
        json.dump(trial_summary, f, indent=2)
    
    # ✅ FIX: Safe best trial check with try-except
    try:
        # Check if this is the best trial so far
        best_trial = trial.study.best_trial
        
        if best_trial is not None and trial.number == best_trial.number:
            best_ckpt_dir = os.path.join(args.optuna_output_dir, "best_trial")
            os.makedirs(best_ckpt_dir, exist_ok=True)
            trainer.save_model(best_ckpt_dir)
            
            # Save best hyperparameters
            best_params_path = os.path.join(best_ckpt_dir, "best_hyperparams.json")
            with open(best_params_path, "w") as f:
                json.dump({
                    "trial_number": trial.number,
                    "mae": final_mae,
                    "rmse": final_rmse,
                    "accuracy": final_acc,
                    "hyperparameters": trial.params,
                }, f, indent=2)
            
            print(f"\n🏆 NEW BEST TRIAL! Saved to {best_ckpt_dir}")
            print(f"   MAE: {final_mae:.4f} (previous best or first trial)")
            
    except (ValueError, AttributeError) as e:
        # ✅ Happens when:
        # 1. This is trial 0 (no completed trials yet in DB)
        # 2. Database is empty/corrupted
        # 3. No successful trials yet
        
        if trial.number == 0:
            # First trial - always save as "current best"
            print(f"\n🥇 Trial 0 complete - saving as initial best")
            best_ckpt_dir = os.path.join(args.optuna_output_dir, "best_trial")
            os.makedirs(best_ckpt_dir, exist_ok=True)
            trainer.save_model(best_ckpt_dir)
            
            best_params_path = os.path.join(best_ckpt_dir, "best_hyperparams.json")
            with open(best_params_path, "w") as f:
                json.dump({
                    "trial_number": trial.number,
                    "mae": final_mae,
                    "rmse": final_rmse,
                    "accuracy": final_acc,
                    "hyperparameters": trial.params,
                }, f, indent=2)
        else:
            print(f"\n⚠️  Could not check best trial status (error: {e})")
            print(f"   This is likely safe to ignore - trial metrics were recorded")
    
    return final_mae


# =============================================================================
# Main: Create Optuna study and run trials
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for StageAB Model")
    
    # Model paths
    parser.add_argument("--llm-model-path", type=str, default="../FlanT5-base", help="Path to Flan-T5 model")
    parser.add_argument("--emb-model-path", type=str, default="../bge-base-en-v1.5", help="Path to BGE model")
    
    # Data paths
    parser.add_argument("--train-file", type=str, default="../LaMP_time_3/train_aug_input.json")
    parser.add_argument("--eval-file", type=str, default="../LaMP_time_3/dev_profile.json")
    parser.add_argument("--graph-dir", type=str, default="../graph_emb")
    parser.add_argument("--bge-emb-dir", type=str, default="../bge_emb")
    
    # Task config
    parser.add_argument("--task-id", type=int, default=3)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-new-len", type=int, default=10)
    parser.add_argument("--max-his-len", type=int, default=10)
    parser.add_argument("--max-session-len", type=int, default=5)
    
    # Training config
    parser.add_argument("--num-epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=50)
    
    # Optuna config
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--optuna-output-dir", type=str, default="./optuna_trials")
    parser.add_argument("--study-name", type=str, default="stageab_fast_hpo")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    
    # Feature toggles
    parser.add_argument("--use-profile", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--use-session", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--use-graph", type=lambda x: str(x).lower() == 'true', default=True)
    
    # Speed hacks (optional)
    parser.add_argument("--subsample-train", type=int, default=0, help="Subsample train set (0=disabled)")
    parser.add_argument("--subsample-eval", type=int, default=0, help="Subsample eval set (0=disabled)")
    
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Create Optuna study with AGGRESSIVE pruning
    # -------------------------------------------------------------------------
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",  # Minimize MAE
        load_if_exists=True,   # Resume if study exists
        
        sampler=TPESampler(
            seed=42,
            n_startup_trials=5,  # Random search for first 5 trials (baseline)
        ),
        
        pruner=MedianPruner(
            n_startup_trials=3,      # Don't prune first 3 trials (need baseline)
            n_warmup_steps=1,         # Start pruning after epoch 1
            interval_steps=1,         # Check every epoch
        ),
    )
    
    print(f"\n{'='*70}")
    print(f"🔬 OPTUNA STUDY: {args.study_name}")
    print(f"{'='*70}")
    print(f"  Direction       : minimize MAE")
    print(f"  Sampler         : TPE (Tree-structured Parzen Estimator)")
    print(f"  Pruner          : MedianPruner (aggressive)")
    print(f"  Total trials    : {args.n_trials}")
    print(f"  Epochs per trial: {args.num_epochs}")
    print(f"  Output dir      : {args.optuna_output_dir}")
    print(f"  Storage         : {args.storage or 'in-memory'}")
    print(f"{'='*70}\n")
    
    # -------------------------------------------------------------------------
    # Run optimization
    # -------------------------------------------------------------------------
    
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print(f"🏆 OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n📊 Best Trial:")
    print(f"  Number : {study.best_trial.number}")
    print(f"  MAE    : {study.best_trial.value:.4f}")
    print(f"\n  Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key:20s}: {value}")
    
    # Save study summary
    summary_path = os.path.join(args.optuna_output_dir, "study_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "study_name": args.study_name,
            "n_trials": len(study.trials),
            "best_trial_number": study.best_trial.number,
            "best_mae": study.best_trial.value,
            "best_params": study.best_trial.params,
        }, f, indent=2)
    
    print(f"\n💾 Study summary saved to: {summary_path}")
    print(f"🏆 Best model checkpoint: {os.path.join(args.optuna_output_dir, 'best_trial')}")
    
    # -------------------------------------------------------------------------
    # Generate visualization (requires optuna-dashboard or matplotlib)
    # -------------------------------------------------------------------------
    try:
        import optuna.visualization as vis
        import plotly
        
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(os.path.join(args.optuna_output_dir, "optimization_history.html"))
        
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html(os.path.join(args.optuna_output_dir, "param_importances.html"))
        
        print(f"\n📈 Visualizations saved to {args.optuna_output_dir}")
    except ImportError:
        print(f"\n⚠️  Install 'plotly' for visualizations: pip install plotly")


if __name__ == "__main__":
    main()