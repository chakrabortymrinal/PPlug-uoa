import os
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn
import torch.nn.functional as F


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
        session_num_layers: int = 2,
        session_num_heads: int = 4,
        cross_num_heads: int = 8,
        graph_dir: str = "../graph_emb",
        bge_emb_dir: str = "../bge_emb",
    ):
        super().__init__()
        self.llm_model = llm_model
        self.emb_model = emb_model
        self.llm_tokenizer = llm_tokenizer

        self.max_input_len = max_input_len
        self.max_new_len = max_new_len
        self.task_id = task_id

        self.bge_emb_dir = bge_emb_dir

        # A/B toggles (must match main_profile flags)
        self.use_profile = bool(use_profile)
        self.use_session = bool(use_session)
        self.use_graph = bool(use_graph)

        # internal toggles (kept ON in main_profile)
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
        self.emb_emb_size = getattr(self.emb_model.config, "hidden_size", None)
        if self.emb_emb_size is None:
            # Very defensive; most HF encoder models have config.hidden_size
            self.emb_emb_size = self.emb_model.get_input_embeddings().embedding_dim

        # ---------------------------------------------------------------------
        # Freeze base models (matches current project approach)
        # ---------------------------------------------------------------------
        for p in self.llm_model.parameters():
            p.requires_grad = False
        for p in self.emb_model.parameters():
            p.requires_grad = False

        # ---------------------------------------------------------------------
        # Special instruction/personalization token (optional)
        # Make it live in EMB space (E), then align with align_mlp_inst -> H.
        # ---------------------------------------------------------------------
        if self.use_inst_token:
            self.inst_token = nn.Parameter(torch.zeros(1, 1, self.emb_emb_size))
            nn.init.normal_(self.inst_token, mean=0.0, std=0.02)
        else:
            self.inst_token = None

        # ---------------------------------------------------------------------
        # Align modules: map BGE/graph embedding space -> LLM hidden size
        # ---------------------------------------------------------------------
        def _make_align():
            return nn.Sequential(
                nn.Linear(self.emb_emb_size, self.llm_emb_size),
                nn.GELU(),
                nn.Linear(self.llm_emb_size, self.llm_emb_size),
            )

        self.align_mlp_inst = _make_align() if self.use_align_mlp_inst else None
        self.align_mlp = _make_align() if self.use_align_mlp else None
        self.align_mlp_session = _make_align() if self.use_align_mlp_session else None
        self.align_mlp_graph = _make_align() if self.use_align_mlp_graph else None

        # ---------------------------------------------------------------------
        # Stage A: history embeddings (memmap, like existing slim pipeline)
        # We support either:
        #   - memmap files in offline_cache_lamp3/
        #   - torch .emb files in bge_emb/
        # Without changing dataset, his_id is already "row indices" into these tables.
        # ---------------------------------------------------------------------
        self.his_train_memmap = None
        self.his_dev_memmap = None
        self.his_train_tensor = None
        self.his_dev_tensor = None
        self._init_history_memmaps(self.bge_emb_dir)

        # ---------------------------------------------------------------------
        # Stage A: graph embeddings (precomputed offline): task_{id}_graph.npy
        # The dataset provides graph_node_ids (pad = -1) and graph_node_mask.
        # ---------------------------------------------------------------------
        self.graph_node_emb = None
        self._init_graph_embeddings(graph_dir)

        # ---------------------------------------------------------------------
        # Stage A: session encoder (Transformer over aligned session tokens)
        # session_ids are history-row ids; we reuse the same history embedding table.
        # ---------------------------------------------------------------------
        self.session_pos_enc = SinusoidalPositionalEncoding(self.llm_emb_size, max_len=1024)
        if self.use_session_encoder:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.llm_emb_size,
                nhead=session_num_heads,
                batch_first=True,
                norm_first=True,
            )
            self.session_encoder = nn.TransformerEncoder(enc_layer, num_layers=session_num_layers)
        else:
            self.session_encoder = None

        # ---------------------------------------------------------------------
        # Stage B: gated cross attention fusion
        # task tokens (queries) attend to user tokens (keys/values)
        # ---------------------------------------------------------------------
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.llm_emb_size,
                num_heads=cross_num_heads,
                batch_first=True,
            )
        else:
            self.cross_attn = None

        # Gate outputs a per-token scalar gate in [0,1]
        if self.use_gate:
            self.gate = nn.Linear(self.llm_emb_size * 2, 1)
            nn.init.constant_(self.gate.bias, -1.0)  # conservative start: prefer task tokens
            self.gate_temperature = nn.Parameter(torch.tensor(1.0))
        else:
            self.gate = None
            self.gate_temperature = None

        # ---------------------------------------------------------------------
        # Diagnostics buffers expected by callbacks / your earlier question
        # ---------------------------------------------------------------------
        self.max_grad_norm = 1.0
        self.grad_norm_check_steps = 50
        self.warmup_steps = 100
        self.loss_ema_alpha = 0.05

        self._loss_ema = None
        self._grad_norms = []  # <-- your earlier question: store grad norms over time
        self._gate_means = []

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate available sources into USER_TOKENS (B, U, H) and key_padding_mask (B, U).
        """
        toks = []
        pads = []

        if self.use_inst_token and self.inst_token is not None:
            inst_emb = self.inst_token.expand(batch_size, -1, -1).to(device=device)  # (B, 1, E)

            if self.align_mlp_inst is not None:
                inst = self.align_mlp_inst(inst_emb)  # (B, 1, H)
            else:
                inst = inst_emb  # assumes E==H (unlikely)

            toks.append(inst)
            pads.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))

        if self.use_profile and his_tokens is not None:
            toks.append(his_tokens)
            pads.append(his_pad)

        if self.use_session and sess_tokens is not None:
            toks.append(sess_tokens)
            pads.append(sess_pad)

        if self.use_graph and graph_tokens is not None:
            toks.append(graph_tokens)
            pads.append(graph_pad)

        if not toks:
            # no personalization sources active: return a dummy 1-token pad so cross-attn doesn't crash
            dummy = torch.zeros((batch_size, 1, self.llm_emb_size), device=device)
            dummy_pad = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            return dummy, dummy_pad

        user_tokens = torch.cat(toks, dim=1)  # (B, U, H)
        user_pad = torch.cat(pads, dim=1)     # (B, U)
        return user_tokens, user_pad

    def _gated_cross_attention(self, task_tokens: torch.Tensor, user_tokens: torch.Tensor, user_pad_mask: torch.Tensor) -> torch.Tensor:
        """
        task_tokens: (B, T, H)
        user_tokens: (B, U, H)
        user_pad_mask: (B, U) True for PAD/invalid
        """
        if self.cross_attn is None:
            return task_tokens

        attn_out, _ = self.cross_attn(
            query=task_tokens,
            key=user_tokens,
            value=user_tokens,
            key_padding_mask=user_pad_mask,
            need_weights=False,
        )  # (B, T, H)

        if self.gate is None:
            return attn_out

        # Gate per token
        gate_in = torch.cat([task_tokens, attn_out], dim=-1)  # (B, T, 2H)
        temp = torch.clamp(self.gate_temperature, 0.1, 10.0) if self.gate_temperature is not None else 1.0
        g = torch.sigmoid(self.gate(gate_in) / temp)  # (B, T, 1)
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
        user_tokens, user_pad = self._build_user_tokens(
            his_tokens, his_pad, sess_tokens, sess_pad, graph_tokens, graph_pad,
            batch_size=bsz, device=device
        )
        fused_task_tokens = self._gated_cross_attention(task_tokens, user_tokens, user_pad)  # (B, T, H)

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