import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import numpy as np
import json
import torch.nn.functional as F

# ✅ NEW: used to pass fused encoder states into HF generate()
from transformers.modeling_outputs import BaseModelOutput


class PersonalLLM_Slim(nn.Module):
    """
    Slim variant with:
    - Frozen Flan-T5 backbone
    - Memmap-based history and graph embeddings
    - Session-aware Transformer
    - Gated cross-attention fusion applied to ALL encoder tokens
    - ✅ FIXED: Stable gradient flow + gate saturation prevention
    """

    def __init__(self, llm_model, emb_model, max_input_len, max_new_len, task_id,
                 llm_tokenizer=None,
                 use_inst_token=True,
                 use_align_mlp_inst=True,
                 use_align_mlp=True,
                 use_session_encoder=True,
                 use_align_mlp_session=True,
                 use_align_mlp_graph=True,
                 use_cross_attn=True,
                 use_gate=True,
                 use_profile=True,   # ✅ profile/history embeddings ON/OFF
                 use_session=True,   # ✅ session embeddings ON/OFF
                 use_graph=True):    # ✅ graph embeddings ON/OFF
        super().__init__()

        # =====================================================================
        # [LAYER 0] CORE BACKBONE + SHAPES
        # =====================================================================
        self.llm_model = llm_model
        self.emb_model = emb_model
        self.llm_tokenizer = llm_tokenizer
        self.max_input_len = max_input_len
        self.max_new_len = max_new_len

        self.llm_config = self.llm_model.config
        self.llm_emb_size = self.llm_config.hidden_size
        self.emb_config = self.emb_model.config
        self.emb_emb_size = self.emb_config.hidden_size

        # =====================================================================
        # [LAYER 1] CASCADE FLAGS (disable dependent modules if a source is OFF)
        # =====================================================================
        # Rule: If a DATA SOURCE is disabled, disable its dependent MODULES
        if not use_profile:
            use_align_mlp = False  # No profile data → no profile alignment needed

        if not use_session:
            use_session_encoder = False  # No session data → no session encoder
            use_align_mlp_session = False

        if not use_graph:
            use_align_mlp_graph = False  # No graph data → no graph alignment

        # =====================================================================
        # [LAYER 2] STORE FLAGS (data sources + derived module toggles)
        # =====================================================================
        # DATA SOURCES (user's explicit choice)
        self.use_profile = use_profile
        self.use_session = use_session
        self.use_graph = use_graph
        self.use_inst_token = use_inst_token

        # MODULES (derived from above + user choice)
        self.use_align_mlp = use_align_mlp  # Only True if use_profile=True
        self.use_align_mlp_inst = use_align_mlp_inst
        self.use_align_mlp_session = use_align_mlp_session  # Only True if use_session=True
        self.use_align_mlp_graph = use_align_mlp_graph  # Only True if use_graph=True
        self.use_session_encoder = use_session_encoder  # Only True if use_session=True
        self.use_cross_attn = use_cross_attn
        self.use_gate = use_gate

        # =====================================================================
        # [LAYER 3] VALIDATION / ACTIVE-SOURCE COUNT (for gate input sizing)
        # =====================================================================
        self.n_active_sources = sum([
            self.use_profile,
            self.use_session,
            self.use_graph,
            self.use_inst_token
        ])

        if self.n_active_sources == 0:
            raise ValueError("❌ No fusion sources active! Enable at least one of: profile, session, graph, inst_token")

        print(f"\n✅ FUSION CONFIGURATION")
        print(f"   Active sources: {self.n_active_sources}")
        print(f"   - Profile:     {self.use_profile}")
        print(f"   - Session:     {self.use_session}")
        print(f"   - Graph:       {self.use_graph}")
        print(f"   - Inst Token:  {self.use_inst_token}\n")

        # =====================================================================
        # [LAYER 4] DEBUG / DIAGNOSTICS CONTROLS
        # =====================================================================
        self.enable_cosine_check = True
        self.cosine_check_every = 25
        self.cosine_check_min_n = 16
        self.enable_gate_monitor = True
        self.gate_monitor_every = 10
        self.gate_monitor_min_n = 64

        # =====================================================================
        # [LAYER 5] TRAINING HEALTH TRACKERS (loss EMA, grads, gate stats)
        # =====================================================================
        self._grad_norms = []
        self._loss_history = []
        self._loss_ema = 0.0
        self._gate_activations = []
        self.loss_ema_alpha = 0.1
        self._forward_step = 0
        self.max_grad_norm = 1.0
        self.grad_norm_check_steps = 50
        self.warmup_steps = 1000

        # =====================================================================
        # [LAYER 6] OFFLINE MEMORY MAPS (history + graph)
        #   - history: train/dev BGE memmap
        #   - graph:   optional node embedding memmap
        # =====================================================================
        train_npy_path = f"../bge_emb/task_{task_id}_train_bge.memmap.npy"
        dev_npy_path   = f"../bge_emb/task_{task_id}_dev_bge.memmap.npy"

        if not (os.path.exists(train_npy_path) and os.path.exists(dev_npy_path)):
            raise FileNotFoundError(f"Memmap files missing: {train_npy_path}, {dev_npy_path}")

        # Load meta for train memmap
        meta_path = train_npy_path.replace('_bge.memmap.npy', '_offsets.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        total_vectors = meta["total_vectors"]
        dim = meta["dim"]

        self.his_train_memmap = np.memmap(train_npy_path, mode='r', dtype='float32', shape=(total_vectors, dim))

        # Same fix for dev
        dev_meta_path = dev_npy_path.replace('_bge.memmap.npy', '_offsets.json')
        with open(dev_meta_path, 'r') as f:
            dev_meta = json.load(f)
        dev_total_vectors = dev_meta["total_vectors"]
        dev_dim = dev_meta["dim"]
        self.his_dev_memmap = np.memmap(dev_npy_path, mode='r', dtype='float32', shape=(dev_total_vectors, dev_dim))

        # Graph embeddings
        graph_path = f"../graph_emb/task_{task_id}_graph.npy"
        self.graph_memmap = np.load(graph_path, mmap_mode='r') if os.path.exists(graph_path) else None

        # =====================================================================
        # [LAYER 7] CONDITIONALLY-CREATED TRAINABLE MODULES (ONLY if needed)
        #   - align_mlp (profile)
        #   - inst token + align_mlp_inst
        #   - session encoder + align_mlp_session
        #   - graph aligner
        #   - cross-attn (currently created but not used in forward)
        #   - gate + temperature
        # =====================================================================

        # PROFILE ALIGNMENT (only if profile source is active)
        if self.use_align_mlp:
            self.align_mlp = self._build_alignment_mlp()
            print("   ✅ Created: align_mlp (profile)")
        else:
            print("   ⏭️  Skipped: align_mlp (profile disabled)")

        # INSTRUCTION TOKEN ALIGNMENT
        if self.use_inst_token:
            self.inst_token = nn.Parameter(torch.rand(self.emb_emb_size), requires_grad=True)
            nn.init.normal_(self.inst_token, mean=0.0, std=0.1)
            print("   ✅ Created: inst_token")
        
        if self.use_align_mlp_inst:
            self.align_mlp_inst = self._build_alignment_mlp()
            print("   ✅ Created: align_mlp_inst (instruction token)")
        else:
            print("   ⏭️  Skipped: align_mlp_inst (inst_token disabled)")

        # SESSION ENCODER + ALIGNMENT (only if session source is active)
        if self.use_session_encoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.llm_emb_size,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.1,
                activation='gelu'
            )
            self.session_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.session_pos_enc = nn.Parameter(
                torch.randn(1, 20, self.llm_emb_size) * 0.01,
                requires_grad=True
            )
            print("   ✅ Created: session_encoder + session_pos_enc")
        else:
            print("   ⏭️  Skipped: session_encoder (session disabled)")

        if self.use_align_mlp_session:
            self.align_mlp_session = self._build_alignment_mlp()
            print("   ✅ Created: align_mlp_session (session)")
        else:
            print("   ⏭️  Skipped: align_mlp_session (session disabled)")

        # GRAPH ALIGNMENT (only if graph source is active)
        if self.use_align_mlp_graph:
            self.align_mlp_graph = self._build_alignment_mlp()
            print("   ✅ Created: align_mlp_graph (graph)")
        else:
            print("   ⏭️  Skipped: align_mlp_graph (graph disabled)")

        # CROSS-ATTENTION (created but not used in current forward)
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.llm_emb_size,
                num_heads=8,
                batch_first=True,
                dropout=0.1
            )
            print("   ✅ Created: cross_attn")
        else:
            print("   ⏭️  Skipped: cross_attn")

        # GATE (with proper input dimension based on active sources)
        if self.use_gate and self.n_active_sources > 0:
            fusion_dim = self.n_active_sources * self.llm_emb_size
            
            self.gate = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_dim // 2, 1)
            )
            
            # ✅ CRITICAL: Initialize to OPEN (sigmoid(3.0) ≈ 0.95)
            with torch.no_grad():
                self.gate[-1].bias.fill_(3.0)  # ← START OPEN
                self.gate[-1].weight.normal_(0.0, 0.02)
            
            # ✅ Temperature for stability
            self.gate_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.gate_logit_clamp = 5.0
            
            print(f"   ✅ Created: gate (input_dim={fusion_dim})")
        else:
            print("   ⏭️  Skipped: gate")

        print()  # Blank line for readability

        # =====================================================================
        # [LAYER 8] FREEZE BACKBONES (LLM + embedding model)
        #   - only the fusion modules above remain trainable
        # =====================================================================
        for _, p in self.llm_model.named_parameters():
            p.requires_grad = False

        for _, p in self.emb_model.named_parameters():
            p.requires_grad = False

        print("✅ Froze: llm_model, emb_model\n")

        # =====================================================================
        # [LAYER 9] TRAINABLE PARAMETER SUMMARY
        # =====================================================================
        self._print_trainable_summary()

    # =====================================================================
    # [UTIL] Trainable parameters summary
    # =====================================================================
    def _print_trainable_summary(self):
        """Print summary of trainable parameters."""
        print("\n" + "="*70)
        print("📊 TRAINABLE PARAMETERS SUMMARY")
        print("="*70)

        trainable_modules = {
            "align_mlp": self.align_mlp if hasattr(self, 'align_mlp') else None,
            "align_mlp_inst": self.align_mlp_inst if hasattr(self, 'align_mlp_inst') else None,
            "align_mlp_session": self.align_mlp_session if hasattr(self, 'align_mlp_session') else None,
            "align_mlp_graph": self.align_mlp_graph if hasattr(self, 'align_mlp_graph') else None,
            "session_encoder": self.session_encoder if hasattr(self, 'session_encoder') else None,
            "session_pos_enc": self.session_pos_enc if hasattr(self, 'session_pos_enc') else None,
            "inst_token": self.inst_token if hasattr(self, 'inst_token') else None,
            "gate": self.gate if hasattr(self, 'gate') else None,
            "gate_temperature": self.gate_temperature if hasattr(self, 'gate_temperature') else None,
        }

        total_trainable = 0
        for name, module in trainable_modules.items():
            if module is None:
                print(f"   ⏭️  {name:25s} | NOT CREATED")
            else:
                if isinstance(module, nn.Parameter):
                    n_params = module.numel()
                    print(f"   ✅ {name:25s} | trainable={n_params:,}")
                    total_trainable += n_params
                else:
                    n_params = sum(p.numel() for p in module.parameters())
                    print(f"   ✅ {name:25s} | trainable={n_params:,}")
                    total_trainable += n_params

        total_all = sum(p.numel() for p in self.parameters())
        print("="*70)
        print(f"Total trainable: {total_trainable:,}")
        print(f"Total params:    {total_all:,}")
        print(f"Trainable ratio: {100 * total_trainable / total_all:.2f}%")
        print("="*70 + "\n")

    # =====================================================================
    # [UTIL] Alignment MLP builder
    # =====================================================================
    def _build_alignment_mlp(self):
        """Build a stable alignment MLP with proper initialization."""
        mlp = nn.Sequential(
            nn.Linear(self.emb_emb_size, self.llm_emb_size),
            nn.GELU(),
            nn.Linear(self.llm_emb_size, self.llm_emb_size)
        )
        with torch.no_grad():
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        return mlp

    # =====================================================================
    # [UTIL] Gradient monitoring / clipping (optional, called externally)
    # =====================================================================
    def clip_and_monitor_gradients(self):
        """Monitor and clip gradients for stability."""
        total_norm = 0.0
        param_count = 0

        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        total_norm = total_norm ** 0.5

        if total_norm > self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            warning = f"Gradients clipped (norm={total_norm:.4f} > {self.max_grad_norm})"
        elif total_norm < 1e-7:
            warning = f"⚠️ Gradient norm extremely low: {total_norm:.2e}"
        else:
            warning = None

        self._grad_norms.append(total_norm)
        return {
            "grad_norm": total_norm,
            "param_count": param_count,
            "warning": warning
        }

    # =====================================================================
    # [UTIL] Training diagnostics snapshot (optional)
    # =====================================================================
    def get_training_diagnostics(self):
        """Get EMA loss, gradient statistics, and gate activation metrics."""
        if not self._grad_norms:
            return {"loss_ema": 0.0, "grad_norm_mean": 0.0}

        grad_norms = np.array(self._grad_norms[-100:])
        loss_history = np.array(self._loss_history[-100:])

        diagnostics = {
            "loss_ema": float(self._loss_ema),
            "grad_norm_mean": float(grad_norms.mean()),
            "grad_norm_std": float(grad_norms.std()),
            "grad_norm_max": float(grad_norms.max()),
            "grad_norm_min": float(grad_norms.min()),
        }

        if len(loss_history) > 1:
            diagnostics["loss_improvement_last_100"] = float(loss_history[0] - loss_history[-1])

        if self._gate_activations:
            gate_vals = np.array(self._gate_activations[-100:])
            diagnostics["gate_mean"] = float(gate_vals.mean())
            diagnostics["gate_std"] = float(gate_vals.std())

        return diagnostics

    # =====================================================================
    # [FORWARD] End-to-end flow:
    #   1) Frozen LLM encoder → task token states
    #   2) Frozen embedding model → token states for session/query pooling
    #   3) Build personalization sources (profile/session/graph/inst)
    #   4) Fuse sources → pers_embs
    #   5) Gate blend pers_embs with task signal → fused_task_embs
    #   6) Prepend fused_task_embs to encoder states (augmentation token)
    #   7) Run frozen LLM decoder + LM head
    #   8) Compute training loss (if labels present)
    # =====================================================================
    def forward(self, llm_input_ids, llm_attention_mask, emb_input_ids, emb_attention_mask,
                emb_token_type_ids=None, his_id=None, session_ids=None, graph_node_ids=None,
                graph_node_mask=None, labels=None):
        self._forward_step += 1
        device = llm_input_ids.device

        # ---------------------------------------------------------------------
        # [1] LLM ENCODING (frozen): produce task token-level states
        # ---------------------------------------------------------------------
        with torch.no_grad():
            encoder_outputs = self.llm_model.encoder(
                input_ids=llm_input_ids,
                attention_mask=llm_attention_mask,
                return_dict=True
            )
            task_embs = encoder_outputs.last_hidden_state  # (batch, seq_len, llm_dim)

        # ---------------------------------------------------------------------
        # [2] EMB MODEL FOR AUX SIGNALS (frozen): token states for pooling
        # ---------------------------------------------------------------------
        with torch.no_grad():
            emb_outputs = self.emb_model(
                input_ids=emb_input_ids,
                attention_mask=emb_attention_mask,
                token_type_ids=emb_token_type_ids,
                return_dict=True
            )
            emb_token_states = emb_outputs.last_hidden_state  # (batch, seq_len, emb_dim)

        # ---------------------------------------------------------------------
        # [3] PERSONALIZATION SOURCES (each contributes a (batch, llm_dim) vector)
        # ---------------------------------------------------------------------
        fusion_inputs = []

        # [3A] PROFILE/HISTORY
        if self.use_profile and his_id is not None and his_id.numel() > 0:
            profile_embs = self._get_history_embeddings(his_id)  # (batch, emb_dim)
            if self.use_align_mlp and profile_embs is not None:
                profile_embs = self.align_mlp(profile_embs)  # (batch, llm_dim)
            if profile_embs is not None:
                fusion_inputs.append(profile_embs)

        # [3B] SESSION
        if self.use_session and session_ids is not None and session_ids.numel() > 0:
            session_embs = self._get_session_embeddings(session_ids, emb_token_states)  # (batch, emb_dim)
            if self.use_session_encoder and session_embs is not None:
                session_embs = self._apply_session_encoder(session_embs)  # (batch, llm_dim)
            if self.use_align_mlp_session and session_embs is not None:
                session_embs = self.align_mlp_session(session_embs)  # (batch, llm_dim)
            if session_embs is not None:
                fusion_inputs.append(session_embs)

        # [3C] INSTRUCTION TOKEN (trainable learned vector)
        if self.use_inst_token:
            inst_expanded = self.inst_token.unsqueeze(0).expand(llm_input_ids.size(0), -1)  # (batch, emb_dim)
            if self.use_align_mlp_inst:
                inst_expanded = self.align_mlp_inst(inst_expanded)  # (batch, llm_dim)
            fusion_inputs.append(inst_expanded)

        # [3D] GRAPH
        if self.use_graph and graph_node_ids is not None and graph_node_ids.numel() > 0:
            graph_embs = self._get_graph_embeddings(graph_node_ids, graph_node_mask)  # (batch, emb_dim)
            if self.use_align_mlp_graph and graph_embs is not None:
                graph_embs = self.align_mlp_graph(graph_embs)  # (batch, llm_dim)
            if graph_embs is not None:
                fusion_inputs.append(graph_embs)

        # ---------------------------------------------------------------------
        # [4] SOURCE AGGREGATION → pers_embs
        # ---------------------------------------------------------------------
        if len(fusion_inputs) == 0:
            pers_embs = torch.zeros(llm_input_ids.size(0), self.llm_emb_size, device=device)
            print(f"⚠️ [Step {self._forward_step}] No personalization sources available!")
        else:
            pers_embs = torch.stack(fusion_inputs, dim=0).mean(dim=0)  # (batch, llm_dim)

        # ---------------------------------------------------------------------
        # [5] GATED BLEND (pers_embs vs task signal)
        # ---------------------------------------------------------------------
        if self.use_gate and len(fusion_inputs) > 0:
            # ✅ FIX: always feed a tensor shaped to the gate's declared input size
            # gate expects: (batch, n_active_sources * llm_dim)
            # If some sources were "active" by flags but missing for this sample, pad with zeros.
            padded_sources = list(fusion_inputs)
            while len(padded_sources) < self.n_active_sources:
                padded_sources.append(torch.zeros_like(padded_sources[0]))

            fused_concat = torch.cat(padded_sources, dim=1)  # (batch, n_active_sources * llm_dim)

            gate_logits = self.gate(fused_concat)  # (batch, 1)

            temp = torch.clamp(self.gate_temperature, min=0.1, max=10.0)
            gate_logits_scaled = gate_logits / temp
            gate_logits_clamped = torch.clamp(
                gate_logits_scaled, min=-self.gate_logit_clamp, max=self.gate_logit_clamp
            )

            gate_weight = torch.sigmoid(gate_logits_clamped)  # (batch, 1)
            self._gate_activations.append(gate_weight.detach().mean().item())

            with torch.no_grad():
                gate_saturated_high = (gate_weight > 0.95).float().mean().item()
                gate_saturated_low = (gate_weight < 0.05).float().mean().item()

                if self._forward_step % self.gate_monitor_every == 0:
                    print(
                        f"[GATE-HEALTH] step={self._forward_step} | "
                        f"mean={gate_weight.mean().item():.4f} | "
                        f"temp={temp.item():.4f} | "
                        f"saturated_high={gate_saturated_high:.1%} | "
                        f"saturated_low={gate_saturated_low:.1%}"
                    )

            task_signal = task_embs.mean(dim=1)  # (batch, llm_dim)
            fused_task_embs = gate_weight * pers_embs + (1.0 - gate_weight) * task_signal

            if self._forward_step % 50 == 0:
                print(
                    f"[FUSION] step={self._forward_step} | "
                    f"gate_mean={gate_weight.mean().item():.4f} | "
                    f"pers_norm={pers_embs.norm(dim=1).mean().item():.4f} | "
                    f"task_norm={task_signal.norm(dim=1).mean().item():.4f} | "
                    f"fused_norm={fused_task_embs.norm(dim=1).mean().item():.4f}"
                )
        else:
            task_signal = task_embs.mean(dim=1)
            fused_task_embs = 0.5 * pers_embs + 0.5 * task_signal

            if self._forward_step % 100 == 0:
                print(f"[FUSION] No gating (step={self._forward_step}), using 50/50 task-persona blend")

        # ---------------------------------------------------------------------
        # [6] AUGMENT ENCODER STATES (prepend fused token)
        # ---------------------------------------------------------------------
        fused_expanded = fused_task_embs.unsqueeze(1)  # (batch, 1, llm_dim)
        augmented_embeddings = torch.cat([fused_expanded, task_embs], dim=1)  # (batch, seq_len+1, llm_dim)

        batch_size = llm_input_ids.size(0)
        fused_mask = torch.ones(batch_size, 1, dtype=llm_attention_mask.dtype, device=device)
        augmented_mask = torch.cat([fused_mask, llm_attention_mask], dim=1)  # (batch, seq_len+1)

        # ---------------------------------------------------------------------
        # [7] DECODER PASS (labels→shifted decoder_input_ids during training)
        # ---------------------------------------------------------------------
        if labels is not None:
            decoder_input_ids = self._shift_tokens_right(
                labels,
                self.llm_model.config.pad_token_id,
                self.llm_model.config.decoder_start_token_id
            )
        else:
            decoder_start_token_id = self.llm_model.config.decoder_start_token_id
            decoder_input_ids = torch.full(
                (batch_size, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=device
            )

        decoder_outputs = self.llm_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=augmented_embeddings,
            encoder_attention_mask=augmented_mask,
            return_dict=True
        )
        sequence_output = decoder_outputs.last_hidden_state  # (batch, dec_len, llm_dim)

        # ---------------------------------------------------------------------
        # [8] LM HEAD + LOSS
        # ---------------------------------------------------------------------
        logits = self.llm_model.lm_head(sequence_output)  # (batch, dec_len, vocab)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            self._loss_ema = self.loss_ema_alpha * loss.item() + (1 - self.loss_ema_alpha) * self._loss_ema
            self._loss_history.append(loss.item())

        return {
            "loss": loss,
            "logits": logits,
        }

    # -------------------------------------------------------------------------
    # ✅ NEW: generation that preserves personalization by passing fused encoder states
    # -------------------------------------------------------------------------
    def generate(
        self,
        llm_input_ids=None,
        llm_attention_mask=None,
        emb_input_ids=None,
        emb_attention_mask=None,
        emb_token_type_ids=None,
        his_id=None,
        session_ids=None,
        graph_node_ids=None,
        graph_node_mask=None,
        **gen_kwargs
    ):
        """
        HuggingFace Seq2SeqTrainer calls model.generate() during eval/predict.

        CRITICAL:
          We must NOT call self.llm_model.generate() directly on raw inputs, because
          that bypasses the fusion token we prepend in forward().

        This implementation reproduces forward()'s encoder augmentation, then calls
        T5.generate() with encoder_outputs.
        """
        device = next(self.parameters()).device

        if llm_input_ids is None:
            raise ValueError("generate() requires llm_input_ids")
        if llm_attention_mask is None:
            llm_attention_mask = llm_input_ids.ne(self.llm_model.config.pad_token_id).long()

        llm_input_ids = llm_input_ids.to(device)
        llm_attention_mask = llm_attention_mask.to(device)

        if emb_input_ids is None or emb_attention_mask is None:
            # In some pipelines you may skip emb inputs; fallback to non-personalized generation
            # rather than crashing.
            return self.llm_model.generate(
                input_ids=llm_input_ids,
                attention_mask=llm_attention_mask,
                **gen_kwargs
            )

        emb_input_ids = emb_input_ids.to(device)
        emb_attention_mask = emb_attention_mask.to(device)
        if emb_token_type_ids is not None:
            emb_token_type_ids = emb_token_type_ids.to(device)
        if his_id is not None:
            his_id = his_id.to(device)
        if session_ids is not None:
            session_ids = session_ids.to(device)
        if graph_node_ids is not None:
            graph_node_ids = graph_node_ids.to(device)
        if graph_node_mask is not None:
            graph_node_mask = graph_node_mask.to(device)

        # --- replicate forward() up to augmented_embeddings/augmented_mask ---
        with torch.no_grad():
            encoder_outputs = self.llm_model.encoder(
                input_ids=llm_input_ids,
                attention_mask=llm_attention_mask,
                return_dict=True
            )
            task_embs = encoder_outputs.last_hidden_state

            emb_outputs = self.emb_model(
                input_ids=emb_input_ids,
                attention_mask=emb_attention_mask,
                token_type_ids=emb_token_type_ids,
                return_dict=True
            )
            emb_token_states = emb_outputs.last_hidden_state

        fusion_inputs = []

        if self.use_profile and his_id is not None and his_id.numel() > 0:
            profile_embs = self._get_history_embeddings(his_id)
            if self.use_align_mlp and profile_embs is not None:
                profile_embs = self.align_mlp(profile_embs)
            if profile_embs is not None:
                fusion_inputs.append(profile_embs)

        if self.use_session and session_ids is not None and session_ids.numel() > 0:
            session_embs = self._get_session_embeddings(session_ids, emb_token_states)
            if self.use_session_encoder and session_embs is not None:
                session_embs = self._apply_session_encoder(session_embs)
            if self.use_align_mlp_session and session_embs is not None:
                session_embs = self.align_mlp_session(session_embs)
            if session_embs is not None:
                fusion_inputs.append(session_embs)

        if self.use_inst_token:
            inst_expanded = self.inst_token.unsqueeze(0).expand(llm_input_ids.size(0), -1)
            if self.use_align_mlp_inst:
                inst_expanded = self.align_mlp_inst(inst_expanded)
            fusion_inputs.append(inst_expanded)

        if self.use_graph and graph_node_ids is not None and graph_node_ids.numel() > 0:
            graph_embs = self._get_graph_embeddings(graph_node_ids, graph_node_mask)
            if self.use_align_mlp_graph and graph_embs is not None:
                graph_embs = self.align_mlp_graph(graph_embs)
            if graph_embs is not None:
                fusion_inputs.append(graph_embs)

        if len(fusion_inputs) == 0:
            pers_embs = torch.zeros(llm_input_ids.size(0), self.llm_emb_size, device=device)
        else:
            pers_embs = torch.stack(fusion_inputs, dim=0).mean(dim=0)

        if self.use_gate and len(fusion_inputs) > 0:
            padded_sources = list(fusion_inputs)
            while len(padded_sources) < self.n_active_sources:
                padded_sources.append(torch.zeros_like(padded_sources[0]))
            fused_concat = torch.cat(padded_sources, dim=1)

            gate_logits = self.gate(fused_concat)
            temp = torch.clamp(self.gate_temperature, min=0.1, max=10.0)
            gate_logits_scaled = gate_logits / temp
            gate_logits_clamped = torch.clamp(
                gate_logits_scaled, min=-self.gate_logit_clamp, max=self.gate_logit_clamp
            )
            gate_weight = torch.sigmoid(gate_logits_clamped)

            task_signal = task_embs.mean(dim=1)
            fused_task_embs = gate_weight * pers_embs + (1.0 - gate_weight) * task_signal
        else:
            task_signal = task_embs.mean(dim=1)
            fused_task_embs = 0.5 * pers_embs + 0.5 * task_signal

        fused_expanded = fused_task_embs.unsqueeze(1)
        augmented_embeddings = torch.cat([fused_expanded, task_embs], dim=1)

        batch_size = llm_input_ids.size(0)
        fused_mask = torch.ones(batch_size, 1, dtype=llm_attention_mask.dtype, device=device)
        augmented_mask = torch.cat([fused_mask, llm_attention_mask], dim=1)

        encoder_outputs_for_generate = BaseModelOutput(last_hidden_state=augmented_embeddings)

        return self.llm_model.generate(
            encoder_outputs=encoder_outputs_for_generate,
            attention_mask=augmented_mask,
            **gen_kwargs
        )

    # ========================================
    # Helper Methods
    # ========================================
    def _get_history_embeddings(self, his_id):
        """Fetch history embeddings from memmap."""
        if his_id is None or his_id.size(0) == 0:
            return None

        batch_size = his_id.size(0)
        profile_embs = []

        for i in range(batch_size):
            his_ids = his_id[i]  # (max_his_len,)
            valid_ids = his_ids[his_ids > 0].cpu().numpy()

            if len(valid_ids) == 0:
                # No history for this sample
                profile_embs.append(torch.zeros(self.emb_emb_size, device=his_id.device))
            else:
                try:
                    # Fetch from memmap (use train memmap for now)
                    his_vectors = self.his_train_memmap[valid_ids.astype(int)]  # (n_his, 768)
                    his_tensor = torch.from_numpy(his_vectors).to(his_id.device).float()
                    # Average history
                    profile_emb = his_tensor.mean(dim=0)  # (768,)
                    profile_embs.append(profile_emb)
                except Exception as e:
                    print(f"⚠️ Error fetching history {i}: {e}")
                    profile_embs.append(torch.zeros(self.emb_emb_size, device=his_id.device))

        return torch.stack(profile_embs, dim=0) if profile_embs else None

    def _get_session_embeddings(self, session_ids, emb_token_states):
        """
        session_ids: (batch, max_session_len) with padding=0
        emb_token_states: (batch, seq_len, emb_dim) token embeddings from emb_model

        Returns:
            (batch, emb_dim) pooled session embedding
        """
        batch_size = session_ids.shape[0]
        device = session_ids.device
        dtype = emb_token_states.dtype

        # Pool emb token states → (batch, emb_dim)
        if emb_token_states.dim() == 3:
            emb_pooled = emb_token_states[:, 0, :]
        else:
            emb_pooled = emb_token_states

        # If no memmap, fallback to "current query emb" gated by session presence
        if not hasattr(self, 'his_train_memmap') and not hasattr(self, 'his_dev_memmap'):
            valid_mask = session_ids.ne(0).any(dim=1)
            session_presence = valid_mask.float().unsqueeze(1)
            return emb_pooled * session_presence

        memmap_src = self.his_train_memmap if self.training else self.his_dev_memmap
        session_embs = torch.zeros(batch_size, emb_pooled.shape[1], device=device, dtype=dtype)

        for i in range(batch_size):
            valid_mask = session_ids[i].ne(0)
            valid_ids = session_ids[i][valid_mask]

            if valid_ids.numel() == 0:
                session_embs[i] = emb_pooled[i]
            else:
                session_vecs = []
                for sess_id in valid_ids.cpu().numpy():
                    sess_id_int = int(sess_id)
                    if 0 <= sess_id_int < memmap_src.shape[0]:
                        try:
                            vec = memmap_src[sess_id_int].astype(np.float32)
                            session_vecs.append(vec)
                        except (IndexError, TypeError):
                            pass

                if session_vecs:
                    session_vecs_arr = np.array(session_vecs)
                    session_embs[i] = torch.from_numpy(session_vecs_arr.mean(axis=0)).to(device).float()
                else:
                    session_embs[i] = emb_pooled[i]

        return session_embs

    def _shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def _apply_session_encoder(self, session_embs):
        """Apply transformer encoder to session embeddings."""
        if not self.use_session_encoder or session_embs is None:
            return session_embs

        session_embs = session_embs.unsqueeze(1) + self.session_pos_enc[:, :1, :]
        encoded = self.session_encoder(session_embs)
        return encoded.squeeze(1)

    def _get_graph_embeddings(self, graph_node_ids, graph_node_mask):
        """Fetch graph embeddings from memmap."""
        if graph_node_ids is None or self.graph_memmap is None:
            return None

        batch_size = graph_node_ids.size(0)
        graph_embs = []

        for i in range(batch_size):
            node_ids = graph_node_ids[i]  # (max_nodes,)
            valid_mask = node_ids >= 0

            if valid_mask.sum() > 0:
                valid_ids = node_ids[valid_mask].cpu().numpy()
                try:
                    node_vectors = self.graph_memmap[valid_ids.astype(int)]  # (n_nodes, 768)
                    node_tensor = torch.from_numpy(node_vectors).to(graph_node_ids.device).float()
                    graph_emb = node_tensor.mean(dim=0)  # (768,)
                except Exception as e:
                    print(f"⚠️ Error fetching graph nodes {i}: {e}")
                    graph_emb = torch.zeros(self.emb_emb_size, device=graph_node_ids.device)
            else:
                graph_emb = torch.zeros(self.emb_emb_size, device=graph_node_ids.device)

            graph_embs.append(graph_emb)

        return torch.stack(graph_embs, dim=0) if graph_embs else None