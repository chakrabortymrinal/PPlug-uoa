
"""
Ablation Study: Quantify session & graph contributions
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extention.ModelForPer_distilbert_GNN_rating import PersonalBertRating
from transformers import AutoTokenizer, AutoModel

TASK_ID = 3
ENCODER_PATH = "../distilbert-base-uncased"
DEV_FILE = "../LaMP_time_3_subset_id/dev_profile.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_HIS_LEN = 10
MAX_SESSION_LEN = 5
MAX_GRAPH_NODES = 10


def _load_checkpoint_state(checkpoint_path: str):
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state = load_file(safetensors_path, device=str(DEVICE))
        print(f"  ✅ Loaded checkpoint: {safetensors_path}")
        return state

    if os.path.exists(pytorch_bin_path):
        state = torch.load(pytorch_bin_path, map_location=DEVICE)
        print(f"  ✅ Loaded checkpoint: {pytorch_bin_path}")
        return state

    raise FileNotFoundError(
        f"No model weights found in {checkpoint_path} "
        f"(expected model.safetensors or pytorch_model.bin)"
    )


def _pad_ids(ids, max_len: int, pad_value: int = 0):
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [pad_value] * (max_len - len(ids))
    return ids


def _load_dev_jsonl(path: str):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _enforce_required_weights(missing_keys, require_prefixes, config_name: str):
    missing_set = set(missing_keys)
    for pref in require_prefixes:
        # If any parameter under that prefix is missing => architecture mismatch for this config
        if any(k.startswith(pref) for k in missing_set):
            raise RuntimeError(
                f"[{config_name}] Checkpoint is incompatible: missing weights under '{pref}*'. "
                f"Refusing to evaluate with random init."
            )

def _dev_has_any_graph_fields(dev_data) -> bool:
    for item in dev_data[:50]:
        if "graph_node_id" in item or "graph_node_ids" in item:
            return True
    return False

def evaluate_ablation(checkpoint_path: str, max_samples: int = 100):
    """
    Compare configurations:
      - always: Profile Only, Profile + Session
      - graph configs: only if dev file contains graph_node_id(s)
    """
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)

    dev_data = _load_dev_jsonl(DEV_FILE)
    state = _load_checkpoint_state(checkpoint_path)

    has_graph = _dev_has_any_graph_fields(dev_data)
    if not has_graph:
        print("ℹ️ Dev file has no graph fields (graph_node_id / graph_node_ids). Graph ablations will be skipped.")

    configs = [
        ("Profile Only", {"use_profile": True, "use_session": False, "use_graph": False}),
        ("Profile + Session", {"use_profile": True, "use_session": True, "use_graph": False}),
    ]
    if has_graph:
        configs.extend([
            ("Profile + Graph", {"use_profile": True, "use_session": False, "use_graph": True}),
            ("Profile + Session + Graph (FULL)", {"use_profile": True, "use_session": True, "use_graph": True}),
        ])

    results = {}

    for config_name, config_args in configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print(f"{'='*60}")

        encoder = AutoModel.from_pretrained(ENCODER_PATH).to(DEVICE)
        model = PersonalBertRating(
            encoder_model=encoder,
            task_id=TASK_ID,
            **config_args
        ).to(DEVICE).eval()

        # --- Load only compatible keys, but do not allow “random init silently” for enabled modules ---
        model_sd = model.state_dict()
        filtered = {}
        skipped_shape = []

        for k, v in state.items():
            if k in model_sd:
                if tuple(v.shape) == tuple(model_sd[k].shape):
                    filtered[k] = v
                else:
                    skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

        missing, unexpected = model.load_state_dict(filtered, strict=False)

        print(f"  📦 Keys in checkpoint: {len(state)} | loaded: {len(filtered)}")
        if missing:
            print(f"  ⚠️ Missing keys: {len(missing)} (showing up to 15)")
            for k in list(missing)[:15]:
                print(f"     - {k}")
        if unexpected:
            print(f"  ⚠️ Unexpected keys (after filtering): {len(unexpected)} (showing up to 15)")
            for k in list(unexpected)[:15]:
                print(f"     - {k}")
        if skipped_shape:
            print(f"  ⚠️ Skipped keys due to shape mismatch: {len(skipped_shape)} (showing up to 10)")
            for k, shp_ckpt, shp_model in skipped_shape[:10]:
                print(f"     - {k}: ckpt{shp_ckpt} vs model{shp_model}")

        # ✅ Hard fail if the config *needs* session/graph but weights are missing
        required_prefixes = []
        if config_args.get("use_session", False):
            required_prefixes.append("session_encoder.")
        if config_args.get("use_graph", False):
            required_prefixes.append("graph_encoder.")
        if config_args.get("use_session", False) or config_args.get("use_graph", False):
            required_prefixes.append("cross_modal_fusion.")

        if required_prefixes:
            _enforce_required_weights(missing, required_prefixes, config_name)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for item in dev_data[:max_samples]:
                input_text = item["input"]
                label = int(item["output"]) - 1  # 0-indexed

                his_ids = _pad_ids(item.get("his_id", []), MAX_HIS_LEN, pad_value=0)
                his_id = torch.tensor([his_ids], dtype=torch.long, device=DEVICE)

                tokens = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                    padding=False
                ).to(DEVICE)

                # ✅ Session: dev may store "session_ids" (preferred) or "session_id"
                session_ids = None
                if config_args.get("use_session", False):
                    ses_raw = item.get("session_ids", item.get("session_id", []))
                    ses_ids = _pad_ids(ses_raw, MAX_SESSION_LEN, pad_value=0)
                    session_ids = torch.tensor([ses_ids], dtype=torch.long, device=DEVICE)

                # ✅ Graph: training json uses "graph_node_id" (singular) -> dataset converts to "graph_node_ids"
                graph_node_ids = None
                graph_node_mask = None
                relation_type_ids = None
                if config_args.get("use_graph", False):
                    g_raw = item.get("graph_node_id", item.get("graph_node_ids", []))
                    node_ids = _pad_ids(g_raw, MAX_GRAPH_NODES, pad_value=-1)

                    # Mirror PersonalDataset_profile_GNN_bert_rating.py logic
                    rel_ids = [(nid % 3) if (isinstance(nid, int) and nid > 0) else -1 for nid in node_ids]
                    node_mask = [1.0 if (isinstance(nid, int) and nid > 0) else 0.0 for nid in node_ids]

                    graph_node_ids = torch.tensor([node_ids], dtype=torch.long, device=DEVICE)
                    graph_node_mask = torch.tensor([node_mask], dtype=torch.float32, device=DEVICE)
                    relation_type_ids = torch.tensor([rel_ids], dtype=torch.long, device=DEVICE)

                output = model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    his_id=his_id,
                    session_ids=session_ids,
                    graph_node_ids=graph_node_ids,
                    graph_node_mask=graph_node_mask,
                    relation_type_ids=relation_type_ids,
                    force_dev_bank=True,
                )

                pred = output.logits.argmax(dim=-1).item()
                all_preds.append(pred)
                all_labels.append(label)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        preds_1to5 = all_preds + 1
        labels_1to5 = all_labels + 1

        acc = accuracy_score(all_labels, all_preds)
        rmse = float(np.sqrt(mean_squared_error(labels_1to5, preds_1to5)))
        mae = float(mean_absolute_error(labels_1to5, preds_1to5))

        print(f"  Accuracy: {acc:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAE:      {mae:.4f}")

        results[config_name] = {"acc": float(acc), "rmse": rmse, "mae": mae}

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    for config_name, metrics in results.items():
        print(f"{config_name:40s}  Acc={metrics['acc']:.4f} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory (contains model.safetensors or pytorch_model.bin)")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    evaluate_ablation(args.checkpoint, max_samples=args.max_samples)


if __name__ == "__main__":
    main()