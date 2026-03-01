import os
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FinetuneConfig:
    tune_llm: bool = False
    tune_emb: bool = False
    tune_inst_token: bool = True
    tune_align_mlps: bool = True
    tune_session_encoder: bool = True
    tune_cross_attn: bool = True
    tune_gate: bool = True
    llm_trainable_regex: Optional[str] = None
    emb_trainable_regex: Optional[str] = None


@dataclass
class TrainHyperparams:
    """
    ✅ CENTRALIZED: All training/architecture hyperparameters in one place
    
    Benefits:
    - Easy Optuna integration (all search space in one object)
    - Clear documentation of tunable knobs
    - Version control friendly (git diff shows param changes)
    - Eliminates scattered hardcoded values
    
    ✅ UPDATED: Best hyperparameters from Optuna study (trial 37, MAE=0.45)
    """
    # =========================================================================
    # OPTIMIZER LEARNING RATES
    # =========================================================================
    lr_default: float = 1e-4
    lr_gate: float = 0.000693424310288062
    lr_cross_attn: float = 5.8039428979633884e-05
    lr_align: float = 0.00038210272987863897
    lr_session: float = 0.00019606824039492657
    lr_llm: float = 1e-5
    lr_emb: float = 1e-5
    weight_decay: float = 0.05395115580901312

    # =========================================================================
    # MAIN GATE PARAMETERS (Profile/Session Fusion)
    # =========================================================================
    gate_bias_init: float = -0.32431152068047325
    gate_temperature_init: float = 1.0959691973956611
    gate_temperature_min: float = 0.1
    gate_temperature_max: float = 2.0

    # =========================================================================
    # GRAPH GATE PARAMETERS (Graph-Specific Fusion)
    # =========================================================================
    graph_gate_bias_init: float = -0.6955597228515471
    graph_gate_temperature_init: float = 1.5790821511577782
    graph_gate_boost_weight: float = 0.24392438684045603
    graph_gate_temperature_min: float = 0.1
    graph_gate_temperature_max: float = 2.0

    # =========================================================================
    # GRAPH ATTENTION BOOSTING (Controls Graph Priority)
    # =========================================================================
    graph_attention_boost: float = 2.0
    graph_attention_bias: float = 0.5
    graph_focus_threshold: float = 0.1

    # =========================================================================
    # SESSION ENCODER ARCHITECTURE
    # =========================================================================
    session_num_layers: int = 2
    session_num_heads: int = 4
    session_dropout: float = 0.1
    session_max_len: int = 1024

    # =========================================================================
    # CROSS-ATTENTION ARCHITECTURE
    # =========================================================================
    cross_num_heads: int = 8
    cross_dropout: float = 0.1

    # =========================================================================
    # TRAINING DYNAMICS
    # =========================================================================
    max_grad_norm: float = 1.3392524375506107
    grad_norm_check_steps: int = 50
    warmup_steps: int = 50
    loss_ema_alpha: float = 0.05

    # =========================================================================
    # ALIGNMENT MLP ARCHITECTURE
    # =========================================================================
    align_mlp_hidden_factor: float = 1.0
    align_mlp_dropout: float = 0.1

    # =========================================================================
    # SPECIAL TOKEN PARAMETERS
    # =========================================================================
    inst_token_init_std: float = 0.02

    # =========================================================================
    # DIAGNOSTIC/LOGGING
    # =========================================================================
    log_attention_weights: bool = False
    log_gate_activations: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:, :t, :].to(dtype=x.dtype, device=x.device)


class PersonalLLM_Slim_StageAB(nn.Module):
    """
    Stage A/B architecture aligned with your initial idea:

    Stage A: Context encoders
      - Long-term user behaviour encoder: token-level history embeddings (his_id -> (B, Lh, E))
      - Session-aware encoder: Transformer over recent session token embeddings (session_ids -> (B, Ls, E))
      - Graph embedding encoder: token-level graph node embeddings (graph_node_ids -> (B, Lg, E))
      - User input encoder: T5 encoder hidden states (B, T, H)

    Stage B: Aggregation and Fusion
      - Input-aware personal aggregator: build USER_TOKENS = concat(history, session, graph, inst_token)
      - Gated cross attention: task tokens attend to USER_TOKENS, then gate merges enhanced tokens into task tokens

    Notes:
      - Does NOT change dataset/collator or main_profile flags.
      - Keeps A/B toggles: use_profile/use_session/use_graph.
      - Keeps stability hooks used by callbacks:
          clip_and_monitor_gradients(), get_training_diagnostics()
      - Leaves the underlying Flan-T5 frozen by default (matches existing pattern).
    """
    def __init__(
        self,
        llm_model,
        emb_model,
        llm_tokenizer,
        max_input_len: int,
        max_new_len: int,
        task_id: int,
        use_profile: bool = True,
        use_session: bool = True,
        use_graph: bool = True,
        use_inst_token: bool = True,
        use_align_mlp_inst: bool = True,
        use_align_mlp: bool = True,
        use_align_mlp_session: bool = True,
        use_align_mlp_graph: bool = True,
        use_session_encoder: bool = True,
        use_cross_attn: bool = True,
        use_gate: bool = True,
        graph_dir: str = "../graph_emb",
        bge_emb_dir: str = "../bge_emb",
        finetune: Optional[FinetuneConfig] = None,
        train_hp: Optional[TrainHyperparams] = None,
    ):
        super().__init__()
        self.llm_model = llm_model
        self.emb_model = emb_model
        self.llm_tokenizer = llm_tokenizer

        self.max_input_len = max_input_len
        self.max_new_len = max_new_len
        self.task_id = task_id
        self.bge_emb_dir = bge_emb_dir

        # ✅ CHANGED: Store config objects first
        self.finetune = finetune or FinetuneConfig()
        self.train_hp = train_hp or TrainHyperparams()

        # A/B toggles
        self.use_profile = bool(use_profile)
        self.use_session = bool(use_session)
        self.use_graph = bool(use_graph)

        # Internal toggles
        self.use_inst_token = bool(use_inst_token)
        self.use_align_mlp_inst = bool(use_align_mlp_inst)
        self.use_align_mlp = bool(use_align_mlp)
        self.use_align_mlp_session = bool(use_align_mlp_session)
        self.use_align_mlp_graph = bool(use_align_mlp_graph)
        self.use_session_encoder = bool(use_session_encoder)
        self.use_cross_attn = bool(use_cross_attn)
        self.use_gate = bool(use_gate)

        # Dimensions
        self.llm_emb_size = getattr(self.llm_model.config, "d_model", None) or self.llm_model.get_input_embeddings().embedding_dim
        self.emb_emb_size = getattr(self.emb_model.config, "hidden_size", None) or self.emb_model.get_input_embeddings().embedding_dim

        # Freeze base models
        for p in self.llm_model.parameters():
            p.requires_grad = False
        for p in self.emb_model.parameters():
            p.requires_grad = False

        # ✅ CHANGED: Special token initialization uses train_hp
        if self.use_inst_token:
            self.inst_token = nn.Parameter(torch.zeros(1, 1, self.emb_emb_size))
            nn.init.normal_(self.inst_token, mean=0.0, std=self.train_hp.inst_token_init_std)
        else:
            self.inst_token = None

        # ✅ CHANGED: Alignment MLPs use train_hp
        def _make_align():
            hidden_size = int(self.emb_emb_size * self.train_hp.align_mlp_hidden_factor)
            return nn.Sequential(
                nn.Linear(self.emb_emb_size, hidden_size),
                nn.GELU(),
                nn.Dropout(self.train_hp.align_mlp_dropout),
                nn.Linear(hidden_size, self.llm_emb_size),
            )

        self.align_mlp_inst = _make_align() if self.use_align_mlp_inst else None
        self.align_mlp = _make_align() if self.use_align_mlp else None
        self.align_mlp_session = _make_align() if self.use_align_mlp_session else None
        self.align_mlp_graph = _make_align() if self.use_align_mlp_graph else None

        # History/Graph embeddings (unchanged)
        self.his_train_memmap = None
        self.his_dev_memmap = None
        self.his_train_tensor = None
        self.his_dev_tensor = None
        self._init_history_memmaps(self.bge_emb_dir)

        self.graph_node_emb = None
        self._init_graph_embeddings(graph_dir)

        # ✅ CHANGED: Session encoder uses train_hp
        self.session_pos_enc = SinusoidalPositionalEncoding(
            self.llm_emb_size, 
            max_len=self.train_hp.session_max_len
        )
        
        if self.use_session_encoder:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.llm_emb_size,
                nhead=self.train_hp.session_num_heads,
                dim_feedforward=self.llm_emb_size * 4,
                dropout=self.train_hp.session_dropout,
                batch_first=True,
                norm_first=True,
            )
            self.session_encoder = nn.TransformerEncoder(
                enc_layer, 
                num_layers=self.train_hp.session_num_layers
            )
        else:
            self.session_encoder = None

        # ✅ CHANGED: Cross-attention uses train_hp
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.llm_emb_size,
                num_heads=self.train_hp.cross_num_heads,
                dropout=self.train_hp.cross_dropout,
                batch_first=True,
            )
        else:
            self.cross_attn = None

        # ✅ CHANGED: Gates use train_hp
        if self.use_gate:
            self.gate = nn.Linear(self.llm_emb_size * 2, 1)
            nn.init.constant_(self.gate.bias, self.train_hp.gate_bias_init)
            self.gate_temperature = nn.Parameter(torch.tensor(self.train_hp.gate_temperature_init))
            
            if self.use_graph:
                self.graph_gate = nn.Linear(self.llm_emb_size * 2, 1)
                nn.init.constant_(self.graph_gate.bias, self.train_hp.graph_gate_bias_init)
                self.graph_gate_temperature = nn.Parameter(torch.tensor(self.train_hp.graph_gate_temperature_init))
            else:
                self.graph_gate = None
                self.graph_gate_temperature = None
        else:
            self.gate = None
            self.gate_temperature = None
            self.graph_gate = None
            self.graph_gate_temperature = None

        self._apply_finetune_policy()

        # ✅ CHANGED: Diagnostics use train_hp
        self.max_grad_norm = self.train_hp.max_grad_norm
        self.grad_norm_check_steps = self.train_hp.grad_norm_check_steps
        self.warmup_steps = self.train_hp.warmup_steps
        self.loss_ema_alpha = self.train_hp.loss_ema_alpha

        self._loss_ema = None
        self._grad_norms = []
        self._gate_means = []
        
        # ✅ CHANGED: Conditional attention logging
        if self.train_hp.log_attention_weights:
            self._last_attn_weights = []
        else:
            self._last_attn_weights = None

    # -------------------------------------------------------------------------
    # Finetune / optimizer utilities
    # -------------------------------------------------------------------------
    def _apply_finetune_policy(self) -> None:
        """
        Applies FinetuneConfig by setting requires_grad on submodules/params.

        Notes:
          - By default, base llm_model and emb_model are frozen above.
          - This method selectively unfreezes components based on self.finetune.
        """
        import re

        # Base models
        if self.finetune.tune_llm:
            if self.finetune.llm_trainable_regex:
                pat = re.compile(self.finetune.llm_trainable_regex)
                for name, p in self.llm_model.named_parameters():
                    p.requires_grad = bool(pat.search(name))
            else:
                for p in self.llm_model.parameters():
                    p.requires_grad = True
        else:
            for p in self.llm_model.parameters():
                p.requires_grad = False

        if self.finetune.tune_emb:
            if self.finetune.emb_trainable_regex:
                pat = re.compile(self.finetune.emb_trainable_regex)
                for name, p in self.emb_model.named_parameters():
                    p.requires_grad = bool(pat.search(name))
            else:
                for p in self.emb_model.parameters():
                    p.requires_grad = True
        else:
            for p in self.emb_model.parameters():
                p.requires_grad = False

        # Inst token
        if self.inst_token is not None:
            self.inst_token.requires_grad = bool(self.finetune.tune_inst_token)

        # Wrapper modules
        for m in [self.align_mlp_inst, self.align_mlp, self.align_mlp_session, self.align_mlp_graph]:
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = bool(self.finetune.tune_align_mlps)

        if self.session_encoder is not None:
            for p in self.session_encoder.parameters():
                p.requires_grad = bool(self.finetune.tune_session_encoder)

        if self.cross_attn is not None:
            for p in self.cross_attn.parameters():
                p.requires_grad = bool(self.finetune.tune_cross_attn)

        if self.gate is not None:
            for p in self.gate.parameters():
                p.requires_grad = bool(self.finetune.tune_gate)
        if self.gate_temperature is not None:
            self.gate_temperature.requires_grad = bool(self.finetune.tune_gate)

        # ✅ NEW: Graph gate trainability
        if hasattr(self, 'graph_gate') and self.graph_gate is not None:
            for p in self.graph_gate.parameters():
                p.requires_grad = bool(self.finetune.tune_gate)
        if hasattr(self, 'graph_gate_temperature') and self.graph_gate_temperature is not None:
            self.graph_gate_temperature.requires_grad = bool(self.finetune.tune_gate)

    def get_param_groups(self):
        """
        Returns optimizer param groups with LRs from TrainHyperparams.
        Use this in your Trainer/optimizer creation code to keep all LRs centralized.
        """
        groups = []

        def add(module: Optional[nn.Module], lr: float, wd: Optional[float] = None):
            if module is None:
                return
            params = [p for p in module.parameters() if p.requires_grad]
            if not params:
                return
            groups.append({
                "params": params,
                "lr": float(lr),
                "weight_decay": float(self.train_hp.weight_decay if wd is None else wd),
            })

        # Wrapper modules
        add(self.align_mlp_inst, self.train_hp.lr_align)
        add(self.align_mlp, self.train_hp.lr_align)
        add(self.align_mlp_session, self.train_hp.lr_align)
        add(self.align_mlp_graph, self.train_hp.lr_align)

        add(self.session_encoder, self.train_hp.lr_session)
        add(self.cross_attn, self.train_hp.lr_cross_attn)
        add(self.gate, self.train_hp.lr_gate)
        add(self.graph_gate, self.train_hp.lr_gate)

        # Standalone params
        if self.inst_token is not None and self.inst_token.requires_grad:
            groups.append({"params": [self.inst_token], "lr": float(self.train_hp.lr_default), "weight_decay": 0.0})
        if self.gate_temperature is not None and self.gate_temperature.requires_grad:
            groups.append({"params": [self.gate_temperature], "lr": float(self.train_hp.lr_gate), "weight_decay": 0.0})
        if self.graph_gate_temperature is not None and self.graph_gate_temperature.requires_grad:
            groups.append({"params": [self.graph_gate_temperature], "lr": float(self.train_hp.lr_gate), "weight_decay": 0.0})

        # Base models (only if enabled)
        if self.finetune.tune_llm:
            add(self.llm_model, self.train_hp.lr_llm)
        if self.finetune.tune_emb:
            add(self.emb_model, self.train_hp.lr_emb)

        # Fallback: if something trainable wasn’t covered
        covered = {id(p) for g in groups for p in g["params"]}
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in covered]
        if other:
            groups.append({"params": other, "lr": float(self.train_hp.lr_default), "weight_decay": float(self.train_hp.weight_decay)})

        return groups

    # -------------------------------------------------------------------------
    # Init helpers
    # -------------------------------------------------------------------------
    def _init_history_memmaps(self, bge_emb_dir: str):
        """
        Initialize history embedding storage.

        Aligned with extention/ModelForPer_slim_GNN.py:
          - expects memmap npy files in bge_emb_dir:
              task_{task_id}_train_bge.memmap.npy
              task_{task_id}_dev_bge.memmap.npy
          - expects offsets json files:
              task_{task_id}_train_offsets.json
              task_{task_id}_dev_offsets.json

        Sets:
          self.his_train_memmap, self.his_dev_memmap (np.memmap)
          and also stores shapes/dims for safety.
        """
        train_npy_path = os.path.join(bge_emb_dir, f"task_{self.task_id}_train_bge.memmap.npy")
        dev_npy_path = os.path.join(bge_emb_dir, f"task_{self.task_id}_dev_bge.memmap.npy")

        if not (os.path.exists(train_npy_path) and os.path.exists(dev_npy_path)):
            raise FileNotFoundError(
                f"Memmap files missing. Expected:\n"
                f"  - {train_npy_path}\n"
                f"  - {dev_npy_path}\n"
                f"Set bge_emb_dir correctly or generate embeddings first."
            )

        # ---- Train meta (authoritative for dim + length) ----
        train_meta_path = train_npy_path.replace("_bge.memmap.npy", "_offsets.json")
        if not os.path.exists(train_meta_path):
            raise FileNotFoundError(f"Missing offsets meta file: {train_meta_path}")

        with open(train_meta_path, "r") as f:
            meta = json.load(f)

        total_vectors = int(meta["total_vectors"])
        dim = int(meta["dim"])

        self.his_train_memmap = np.memmap(
            train_npy_path,
            mode="r",
            dtype="float32",
            shape=(total_vectors, dim),
        )

        # ---- Dev meta ----
        dev_meta_path = dev_npy_path.replace("_bge.memmap.npy", "_offsets.json")
        if not os.path.exists(dev_meta_path):
            raise FileNotFoundError(f"Missing offsets meta file: {dev_meta_path}")

        with open(dev_meta_path, "r") as f:
            dev_meta = json.load(f)

        dev_total_vectors = int(dev_meta["total_vectors"])
        dev_dim = int(dev_meta["dim"])

        if dev_dim != dim:
            raise ValueError(f"Train/dev dim mismatch: train_dim={dim}, dev_dim={dev_dim}")

        self.his_dev_memmap = np.memmap(
            dev_npy_path,
            mode="r",
            dtype="float32",
            shape=(dev_total_vectors, dev_dim),
        )

        # Optional: store for debugging/validation
        self.his_emb_dim = dim
        self.his_train_size = total_vectors
        self.his_dev_size = dev_total_vectors

    def _init_graph_embeddings(self, graph_dir: str):
        """
        Load graph node embeddings and register as a non-persistent buffer.

        Fix:
          Avoid KeyError when this init is called more than once (e.g., multiple
          constructors/paths, or re-init during experiments). If the buffer already
          exists, just overwrite it via setattr instead of register_buffer().
        """
        import os
        import numpy as np
        import torch

        graph_path = os.path.join(graph_dir, f"task_{self.task_id}_graph.npy")
        if not os.path.exists(graph_path):
            self.graph_node_emb = None
            return

        arr = np.load(graph_path, mmap_mode="r")

        t = torch.tensor(arr, dtype=torch.float32)

        # ✅ If buffer already registered, do NOT call register_buffer again.
        if "graph_node_emb" in self._buffers:
            self._buffers["graph_node_emb"] = t
        elif hasattr(self, "graph_node_emb"):
            # attribute exists but isn't a buffer (defensive)
            setattr(self, "graph_node_emb", t)
        else:
            self.register_buffer("graph_node_emb", t, persistent=False)

    # -------------------------------------------------------------------------
    # Utilities: gather embeddings
    # -------------------------------------------------------------------------
    def _gather_history_emb(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (B, L) history row indices; padding=0 is treated as "no item"
        returns: (B, L, emb_emb_size) in embedding-model space
        """
        if ids is None:
            raise ValueError("his_id/session_ids are required for history embedding lookup.")

        # 1) Prefer memmap if present
        mem = self.his_train_memmap if self.training else self.his_dev_memmap
        if mem is not None:
            ids_cpu = ids.detach().to("cpu").long().clamp_min(0)
            b, l = ids_cpu.shape
            out = torch.zeros((b, l, mem.shape[1]), dtype=torch.float32)

            flat = ids_cpu.view(-1)
            nz_mask = flat.ne(0)
            nz_ids = flat[nz_mask].numpy()
            nz_ids = np.clip(nz_ids, 0, mem.shape[0] - 1)
            gathered = torch.from_numpy(mem[nz_ids]).to(torch.float32)

            out.view(-1, mem.shape[1])[nz_mask] = gathered
            return out.to(device=ids.device)

        # 2) Fallback to loaded torch tensors (.emb)
        table = self.his_train_tensor if self.training else self.his_dev_tensor
        if table is not None:
            ids_safe = ids.long().clamp_min(0).clamp_max(table.size(0) - 1)
            out = table.index_select(0, ids_safe.view(-1).to("cpu")).view(ids.size(0), ids.size(1), -1)
            return out.to(device=ids.device)

        # 3) Last fallback: zeros
        b, l = ids.shape
        return torch.zeros((b, l, self.emb_emb_size), device=ids.device, dtype=torch.float32)

    def _gather_graph_emb(self, graph_node_ids: torch.Tensor) -> torch.Tensor:
        """
        graph_node_ids: (B, Lg) with pad=-1
        returns: (B, Lg, emb_emb_size) in embedding-model space
        """
        b, l = graph_node_ids.shape
        if self.graph_node_emb is None:
            return torch.zeros((b, l, self.emb_emb_size), device=graph_node_ids.device, dtype=torch.float32)

        ids = graph_node_ids.long()
        # valid: >=0
        valid = ids.ge(0)
        safe = ids.clamp_min(0).clamp_max(self.graph_node_emb.size(0) - 1)

        out = torch.zeros((b, l, self.graph_node_emb.size(1)), device=graph_node_ids.device, dtype=self.graph_node_emb.dtype)
        out[valid] = self.graph_node_emb[safe[valid]].to(device=graph_node_ids.device)
        return out.to(torch.float32)

    # -------------------------------------------------------------------------
    # Stage A encoders
    # -------------------------------------------------------------------------
    def _encode_user_input(self, llm_input_ids: torch.Tensor, llm_attention_mask: torch.Tensor) -> torch.Tensor:
        # (B, T, H)
        enc = self.llm_model.encoder(input_ids=llm_input_ids, attention_mask=llm_attention_mask)
        return enc.last_hidden_state

    def _encode_task_for_relevance(self, emb_input_ids: torch.Tensor, emb_attention_mask: torch.Tensor, emb_token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        BGE encoder pooled output for relevance scoring (B, E).
        We keep this simple (CLS/mean depends on model; we use mean pool by mask).
        """
        out = self.emb_model(input_ids=emb_input_ids, attention_mask=emb_attention_mask, token_type_ids=emb_token_type_ids)
        h = out.last_hidden_state  # (B, S, E)
        mask = emb_attention_mask.unsqueeze(-1).to(h.dtype)  # (B, S, 1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled  # (B, E)

    def _encode_long_term_history_tokens(self, his_id: torch.Tensor, task_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          his_tokens_aligned: (B, Lh, H)
          his_key_padding_mask: (B, Lh) True for PAD positions (for MHA key_padding_mask)
        """
        # gather raw history embeddings in emb space
        his_emb = self._gather_history_emb(his_id)  # (B, Lh, E)
        pad_mask = his_id.eq(0)  # True where pad

        # Input-aware weighting (Stage B-ish) but still token-level:
        # we keep token-level representations, but we can downweight irrelevant items by scaling tokens.
        # score = dot(his_emb, task_emb)
        scores = torch.bmm(his_emb, task_emb.unsqueeze(-1)).squeeze(-1)  # (B, Lh)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, Lh, 1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        # scale tokens (keeps token set, but makes attention prefer relevant ones)
        his_emb = his_emb * w

        # align to LLM space
        if self.align_mlp is not None:
            his_tokens = self.align_mlp(his_emb)  # (B, Lh, H)
        else:
            # if no align, assume dims match (unlikely)
            his_tokens = his_emb

        return his_tokens, pad_mask

    def _encode_session_tokens(self, session_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Session-aware encoder over recent interactions.
        Returns:
          session_tokens: (B, Ls, H)
          session_pad_mask: (B, Ls) True for PAD
        """
        sess_emb = self._gather_history_emb(session_ids)  # (B, Ls, E)
        pad_mask = session_ids.eq(0)

        if self.align_mlp_session is not None:
            sess_tokens = self.align_mlp_session(sess_emb)  # (B, Ls, H)
        else:
            sess_tokens = sess_emb

        # Add positional encoding then transformer encode
        if self.session_encoder is not None:
            sess_tokens = self.session_pos_enc(sess_tokens)
            # key_padding_mask expects True for pads
            sess_tokens = self.session_encoder(sess_tokens, src_key_padding_mask=pad_mask)
        return sess_tokens, pad_mask

    def _encode_graph_tokens(self, graph_node_ids: torch.Tensor, graph_node_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Graph embedding encoder returns token-level node embeddings.
        graph_node_ids pad=-1, and graph_node_mask is 0/1.
        Returns:
          graph_tokens: (B, Lg, H)
          graph_pad_mask: (B, Lg) True for PAD (invalid nodes)
        """
        g_emb = self._gather_graph_emb(graph_node_ids)  # (B, Lg, Eg)
        if graph_node_mask is None:
            # valid if id >= 0
            pad_mask = graph_node_ids.lt(0)
        else:
            pad_mask = graph_node_mask.eq(0)

        if self.align_mlp_graph is not None:
            g_tokens = self.align_mlp_graph(g_emb)  # (B, Lg, H)
        else:
            g_tokens = g_emb
        return g_tokens, pad_mask

    # -------------------------------------------------------------------------
    # Stage B fusion
    # -------------------------------------------------------------------------
    def _build_user_tokens(
        self,
        his_tokens: Optional[torch.Tensor],
        his_pad: Optional[torch.Tensor],
        sess_tokens: Optional[torch.Tensor],
        sess_pad: Optional[torch.Tensor],
        graph_tokens: Optional[torch.Tensor],
        graph_pad: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        ✅ CHANGED: Graph importance now uses train_hp.graph_attention_boost
        """
        toks = []
        pads = []
        graph_mask_parts = []
        
        # ✅ CHANGED: Graph tokens first with configurable boost
        if self.use_graph and graph_tokens is not None:
            toks.append(graph_tokens)
            pads.append(graph_pad)
            graph_mask_parts.append(
                torch.ones_like(graph_pad, dtype=torch.float32) * self.train_hp.graph_attention_boost
            )
        
        if self.use_inst_token and self.inst_token is not None:
            inst_emb = self.inst_token.expand(batch_size, -1, -1).to(device=device)
            if self.align_mlp_inst is not None:
                inst = self.align_mlp_inst(inst_emb)
            else:
                inst = inst_emb
            toks.append(inst)
            pads.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))
            graph_mask_parts.append(torch.ones((batch_size, 1), dtype=torch.float32, device=device))
        
        if self.use_session and sess_tokens is not None:
            toks.append(sess_tokens)
            pads.append(sess_pad)
            graph_mask_parts.append(torch.ones_like(sess_pad, dtype=torch.float32))
        
        if self.use_profile and his_tokens is not None:
            toks.append(his_tokens)
            pads.append(his_pad)
            graph_mask_parts.append(torch.ones_like(his_pad, dtype=torch.float32))
        
        if not toks:
            dummy = torch.zeros((batch_size, 1, self.llm_emb_size), device=device)
            dummy_pad = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            return dummy, dummy_pad, None
        
        user_tokens = torch.cat(toks, dim=1)
        user_pad = torch.cat(pads, dim=1)
        
        graph_importance = torch.cat(graph_mask_parts, dim=1) if (self.use_graph and graph_tokens is not None) else None
        
        return user_tokens, user_pad, graph_importance

    def _gated_cross_attention(
        self, 
        task_tokens: torch.Tensor, 
        user_tokens: torch.Tensor, 
        user_pad_mask: torch.Tensor,
        graph_importance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        ✅ CHANGED: All hyperparameters now from train_hp
        ✅ FIXED: Proper QKV projection handling for PyTorch MultiheadAttention
        """
        if self.cross_attn is None:
            return task_tokens
        
        attn_out, attn_weights = self.cross_attn(
            query=task_tokens,
            key=user_tokens,
            value=user_tokens,
            key_padding_mask=user_pad_mask,
            need_weights=True,
        )
        
        # ✅ CHANGED: Graph attention boosting uses train_hp.graph_attention_bias
        if graph_importance is not None and self.use_graph:
            # ✅ FIXED: Manual QKV projection using in_proj_weight/bias
            # PyTorch MultiheadAttention stores QKV as single matrix when _qkv_same_embed_dim=True
            embed_dim = self.llm_emb_size
            
            if self.cross_attn._qkv_same_embed_dim:
                # Single projection matrix: [3*embed_dim, embed_dim]
                q_proj_weight = self.cross_attn.in_proj_weight[:embed_dim, :]
                k_proj_weight = self.cross_attn.in_proj_weight[embed_dim:2*embed_dim, :]
                v_proj_weight = self.cross_attn.in_proj_weight[2*embed_dim:, :]
                
                if self.cross_attn.in_proj_bias is not None:
                    q_proj_bias = self.cross_attn.in_proj_bias[:embed_dim]
                    k_proj_bias = self.cross_attn.in_proj_bias[embed_dim:2*embed_dim]
                    v_proj_bias = self.cross_attn.in_proj_bias[2*embed_dim:]
                else:
                    q_proj_bias = k_proj_bias = v_proj_bias = None
                
                Q = F.linear(task_tokens, q_proj_weight, q_proj_bias)
                K = F.linear(user_tokens, k_proj_weight, k_proj_bias)
                V = F.linear(user_tokens, v_proj_weight, v_proj_bias)
            else:
                # Separate projections (fallback, rarely used)
                Q = F.linear(task_tokens, self.cross_attn.q_proj_weight, getattr(self.cross_attn, 'q_proj_bias', None))
                K = F.linear(user_tokens, self.cross_attn.k_proj_weight, getattr(self.cross_attn, 'k_proj_bias', None))
                V = F.linear(user_tokens, self.cross_attn.v_proj_weight, getattr(self.cross_attn, 'v_proj_bias', None))
            
            num_heads = self.cross_attn.num_heads
            head_dim = embed_dim // num_heads
            
            # Reshape to multi-head format: [B, T, H] -> [B, num_heads, T, head_dim]
            Q = Q.view(Q.size(0), Q.size(1), num_heads, head_dim).transpose(1, 2)
            K = K.view(K.size(0), K.size(1), num_heads, head_dim).transpose(1, 2)
            V = V.view(V.size(0), V.size(1), num_heads, head_dim).transpose(1, 2)
            
            # Scaled dot-product attention with graph bias
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            
            # ✅ CHANGED: Use centralized bias parameter
            graph_bias = (graph_importance - 1.0).unsqueeze(1).unsqueeze(1) * self.train_hp.graph_attention_bias
            scores = scores + graph_bias
            
            if user_pad_mask is not None:
                scores = scores.masked_fill(
                    user_pad_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            
            attn_probs = F.softmax(scores, dim=-1)
            attn_out_custom = torch.matmul(attn_probs, V)
            
            # Reshape back: [B, num_heads, T, head_dim] -> [B, T, embed_dim]
            attn_out_custom = attn_out_custom.transpose(1, 2).contiguous().view(
                attn_out_custom.size(0), attn_out_custom.size(2), embed_dim
            )
            
            # Apply output projection
            attn_out = self.cross_attn.out_proj(attn_out_custom)
            
            # ✅ CHANGED: Conditional logging
            if self.train_hp.log_attention_weights:
                if self._last_attn_weights is None:
                    self._last_attn_weights = []
                self._last_attn_weights.append(attn_probs.detach())
        
        if self.gate is None:
            return attn_out
        
        # ✅ CHANGED: Use centralized temperature bounds
        gate_in = torch.cat([task_tokens, attn_out], dim=-1)
        temp = torch.clamp(
            self.gate_temperature,
            self.train_hp.gate_temperature_min,
            self.train_hp.gate_temperature_max,
        )
        g = torch.sigmoid(self.gate(gate_in) / temp)
        
        # ✅ CHANGED: Graph gate uses centralized parameters
        if self.use_graph and self.graph_gate is not None:
            graph_boost_in = torch.cat([task_tokens, attn_out], dim=-1)
            
            graph_temp = torch.clamp(
                self.graph_gate_temperature,
                self.train_hp.graph_gate_temperature_min,
                self.train_hp.graph_gate_temperature_max
            )
            
            graph_boost = torch.sigmoid(self.graph_gate(graph_boost_in) / graph_temp)
            
            if graph_importance is not None:
                graph_focus = (attn_weights * graph_importance.unsqueeze(1)).sum(dim=-1, keepdim=True)
                graph_focus = graph_focus / (graph_importance.sum(dim=-1, keepdim=True).unsqueeze(1) + 1e-8)
                
                # ✅ CHANGED: Use centralized threshold
                graph_boost = graph_boost * (graph_focus > self.train_hp.graph_focus_threshold).float()
            
            # ✅ CHANGED: Use centralized boost weight
            g = torch.clamp(g + graph_boost * self.train_hp.graph_gate_boost_weight, 0.0, 1.0)
        
        # ✅ CHANGED: Conditional gate logging
        if self.train_hp.log_gate_activations:
            self._gate_means.append(float(g.mean().detach().cpu()))
        
        fused = (1.0 - g) * task_tokens + g * attn_out
        return fused

    # -------------------------------------------------------------------------
    # HF Trainer interface
    # -------------------------------------------------------------------------
    def forward(
        self,
        llm_input_ids: torch.Tensor,
        llm_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        emb_input_ids: Optional[torch.Tensor] = None,
        emb_attention_mask: Optional[torch.Tensor] = None,
        emb_token_type_ids: Optional[torch.Tensor] = None,
        his_id: Optional[torch.Tensor] = None,
        session_ids: Optional[torch.Tensor] = None,
        graph_node_ids: Optional[torch.Tensor] = None,
        graph_node_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        device = llm_input_ids.device
        bsz = llm_input_ids.size(0)

        # Stage A: user input encoding (task tokens)
        task_tokens = self._encode_user_input(llm_input_ids, llm_attention_mask)  # (B, T, H)

        # Task embedding for relevance scoring (BGE pooled)
        if emb_input_ids is None:
            # fallback: zeros
            task_emb = torch.zeros((bsz, self.emb_emb_size), device=device, dtype=torch.float32)
        else:
            if emb_token_type_ids is None:
                emb_token_type_ids = torch.zeros_like(emb_input_ids)
            with torch.no_grad():
                task_emb = self._encode_task_for_relevance(emb_input_ids, emb_attention_mask, emb_token_type_ids)  # (B, E)

        # Stage A: context encoders -> token sets
        his_tokens = his_pad = None
        if self.use_profile and his_id is not None:
            his_tokens, his_pad = self._encode_long_term_history_tokens(his_id, task_emb)

        sess_tokens = sess_pad = None
        if self.use_session and session_ids is not None:
            sess_tokens, sess_pad = self._encode_session_tokens(session_ids)

        graph_tokens = graph_pad = None
        if self.use_graph and graph_node_ids is not None:
            graph_tokens, graph_pad = self._encode_graph_tokens(graph_node_ids, graph_node_mask)

        # Stage B: build user token memory + cross-attn fusion
        user_tokens, user_pad, graph_importance = self._build_user_tokens(
            his_tokens, his_pad, sess_tokens, sess_pad, graph_tokens, graph_pad,
            batch_size=bsz, device=device
        )
        fused_task_tokens = self._gated_cross_attention(task_tokens, user_tokens, user_pad, graph_importance)  # (B, T, H)

        # Feed fused encoder states into T5 decoder via encoder_outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=fused_task_tokens)

        out = self.llm_model(
            encoder_outputs=encoder_outputs,
            attention_mask=llm_attention_mask,
            labels=labels,
            use_cache=False,
        )

        loss = out.loss
        if loss is not None:
            val = float(loss.detach().cpu())
            if self._loss_ema is None:
                self._loss_ema = val
            else:
                self._loss_ema = (1.0 - self.loss_ema_alpha) * self._loss_ema + self.loss_ema_alpha * val

        return {"loss": out.loss, "logits": out.logits}

    # -------------------------------------------------------------------------
    # Callbacks support
    # -------------------------------------------------------------------------
    def clip_and_monitor_gradients(self) -> Dict[str, Any]:
        """
        Called by GradientMonitoringCallback in main_profile-slim-GNN.py.
        Clips gradients and returns grad norm.
        """
        params = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
        if not params:
            grad_norm = 0.0
        else:
            grad_norm = float(torch.norm(torch.stack([p.grad.detach().data.norm(2) for p in params]), 2).detach().cpu())

        # Clip (Trainer will also clip, but we keep this for your callback)
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(getattr(self, "max_grad_norm", 1.0)))

        self._grad_norms.append(grad_norm)

        warning = None
        if grad_norm != grad_norm:  # NaN
            warning = "Gradient norm is NaN (check inputs/initialization)."
        elif grad_norm > 50.0:
            warning = "Gradient norm is extremely large (possible instability)."
        elif 0.0 < grad_norm < 1e-10:
            warning = "Gradient norm is extremely small (model may be effectively frozen)."

        return {"grad_norm": grad_norm, "warning": warning}

    def get_training_diagnostics(self) -> Dict[str, Any]:
        gmean = float(np.mean(self._grad_norms)) if self._grad_norms else 0.0
        gate_mean = float(np.mean(self._gate_means)) if self._gate_means else 0.0
        return {
            "loss_ema": float(self._loss_ema) if self._loss_ema is not None else 0.0,
            "grad_norm_mean": gmean,
            "gate_mean": gate_mean,
        }