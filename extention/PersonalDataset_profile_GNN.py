import copy
import json
import sys
import torch
import os

class PersonalDataset:
    """
    Dataset class for Personalized LLM training with:
      (A) Long-term history      -> `his_id`        (padded to max_his_len, padding=0)
      (B) Short-term session     -> `session_ids`   (recent slice of history, padded to max_session_len)
      (C) Graph neighborhood     -> `graph_node_ids` + `graph_node_mask`
                                  (mapped from his_id using a JSON dict; missing -> -1)

    Design goals:
      - Avoid duplicate signals:
          `session_ids` is derived from the LAST N elements of `his_id_list`
      - Keep IDs safe for memmap indexing:
          Optional clamp with `max_valid_his_id`
      - Keep graph IDs maskable:
          Use -1 for "missing node" (so 0 can remain a valid node if needed)

    ✅ CUDA COMPLIANCE:
      - All tensors returned from __getitem__() are on CPU (PyTorch Dataset convention)
      - DataLoader/collator handles batching and device transfer (.to(device))
      - No GPU operations in Dataset class (prevents multiprocessing issues)

    Expected JSONL record structure:
      {
        "input":  "<question/prompt>",
        "output": "<label/target text>",
        "his_id": [<history_id_1>, <history_id_2>, ...],
        "id":     "<example id>"
      }

    Output dict (per sample) for the model forward():
      {
        "llm_input_ids":       (seq,)  tokenized + special personalization tokens appended
        "llm_attention_mask":  (seq,)  attention mask for llm_input_ids
        "labels":              (tgt,)  token ids for target output
        "emb_input_ids":       (seq,)  token ids for embedding model
        "emb_attention_mask":  (seq,)  attention mask for embedding model
        "emb_token_type_ids":  (seq,)  (zeros; kept for model API compatibility)
        "his_id":              (max_his_len,)
        "session_ids":         (max_session_len,)
        "graph_node_ids":      (max_his_len,)
        "graph_node_mask":     (max_his_len,)  1 where node exists else 0
      }
    """
    def __init__(self, data_file, max_input_len, max_new_len, max_his_len,
                 llm_tokenizer, emb_tokenizer, graph_emb_path=None,
                 his_to_graph_path=None, max_session_len=3, max_valid_his_id=None):

        # =====================================================================
        # [CFG 0] BASIC CONFIG
        # =====================================================================
        self.data_file = data_file
        self.max_input_len = max_input_len
        self.max_new_len = max_new_len
        self.max_his_len = max_his_len

        # NOTE: this code enforces a *minimum* session length of 8.
        # If you pass max_session_len < 8, it will still use 8.
        self.max_session_len = max(max_session_len, 8)  # ✅ Enforce minimum of 8 for recency modeling

        # Optional upper bound for memmap indexing safety (history embedding tables)
        self.max_valid_his_id = max_valid_his_id

        # Tokenizers:
        #   - llm_tokenizer: used to build Flan-T5 input ids + labels
        #   - emb_tokenizer: used to build BGE (embedding model) input ids
        self.llm_tokenizer = llm_tokenizer
        self.emb_tokenizer = emb_tokenizer

        # Graph config (mapping + optional embedding path)
        self.graph_emb_path = graph_emb_path
        self.his_to_graph_path = his_to_graph_path

        # Prevent HF fast-tokenizers from auto-clamping to a default max length
        self.llm_tokenizer.model_max_length = sys.maxsize
        self.emb_tokenizer.model_max_length = sys.maxsize

        # =====================================================================
        # [CFG 1] OPTIONAL his_id → graph_node_id MAPPING
        #   - This mapping is produced by your graph pipeline
        #   - It allows the dataset to emit graph node IDs aligned with history
        # =====================================================================
        self.his_to_graph = {}
        if his_to_graph_path and os.path.exists(his_to_graph_path):
            try:
                with open(his_to_graph_path, "r") as f:
                    self.his_to_graph = json.load(f)
                print(f"📂 Loaded his_to_graph mapping ({len(self.his_to_graph)} entries)")
            except Exception as e:
                print(f"⚠️ Could not load his_to_graph mapping: {e}")

        # =====================================================================
        # [CFG 2] LOAD DATA (JSONL)
        # =====================================================================
        # Stored as raw lines and parsed on demand in __getitem__
        with open(self.data_file, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        """Number of records in the dataset."""
        return len(self.lines)

    # =====================================================================
    # [UTIL] Normalize history IDs to the mapping's key format
    # =====================================================================
    def normalize_his_id(self, hid):
        """
        Convert a history id `hid` to a key that exists in `self.his_to_graph`.

        This repo has seen several key styles in mapping files, e.g.:
          - "12345"
          - "review_12345"
          - keys that end with the numeric id (suffix match)

        Returns:
            normalized_key (str) if a match is found, else None
        """
        key = str(hid)

        # Direct match
        if key in self.his_to_graph:
            return key

        # Prefixed review key match
        review_key = f"review_{hid}"
        if review_key in self.his_to_graph:
            return review_key

        # Suffix match fallback for shorter IDs
        if len(key) < 7:
            for k in self.his_to_graph.keys():
                if isinstance(k, str) and k.endswith(key):
                    return k

        return None
    
    # =====================================================================
    # [UTIL] Pad/truncate variable-length ID lists to a fixed tensor
    # =====================================================================
    def pad_his(self, his_ids, pad_to_len=None):
        """
        Pad or truncate a list of IDs to a fixed length.

        ✅ CUDA COMPLIANCE: Returns CPU tensor (device placement handled by collator).

        Padding conventions:
          - history / session ids: 0 means "pad / missing"
          - graph_node_ids:        -1 means "missing node" BEFORE padding;
                                 padding is still 0 here because we reuse pad_his,
                                 but we also emit `graph_node_mask` so model can ignore.

        Args:
            his_ids: list[int]
            pad_to_len: target length (defaults to self.max_his_len)

        Returns:
            torch.LongTensor of shape (pad_to_len,) on CPU
        """
        pad_len = pad_to_len if pad_to_len is not None else self.max_his_len
        his_ids = his_ids[:pad_len]                      # truncate
        his_ids += [0] * (pad_len - len(his_ids))        # pad with zeros
        
        # ✅ CUDA FIX: Explicitly create on CPU (default, but being explicit)
        return torch.tensor(his_ids, dtype=torch.long, device='cpu')

    # =====================================================================
    # [GETITEM] Build one sample dict consumed by Trainer/DataCollator
    # =====================================================================
    def __getitem__(self, idx):
        """
        Build a single training sample dict for DataLoader.

        ✅ CUDA COMPLIANCE:
          - All tensors returned on CPU (PyTorch Dataset convention)
          - DataLoader will batch these and move to device
          - Model's forward() receives tensors on correct device

        High-level stages:
          [1] Parse JSONL → input_str, output_str, raw his_id_list
          [2] Sanitize history IDs (int cast + optional clamp)
          [3] Construct:
              - his_id      (long-term, padded to max_his_len)
              - session_ids (recent slice, padded to max_session_len)
              - graph_node_ids + graph_node_mask (aligned with history)
          [4] Tokenize input for:
              - LLM backbone (T5) + append special personalization tokens
              - Embedding model (BGE)
          [5] Tokenize output → labels
        """
        # ---------------------------------------------------------------------
        # [1] Parse single JSONL record
        # ---------------------------------------------------------------------
        input_str, output_str, his_id_list = self.parse_data(self.lines[idx])

        # ---------------------------------------------------------------------
        # [2] Sanitize history IDs
        #   - cast to int (non-castable -> 0)
        #   - optional clamp to keep indices within memmap size
        # ---------------------------------------------------------------------
        safe_his_ids = []
        for hid in his_id_list:
            try:
                v = int(hid)
            except (TypeError, ValueError):
                v = 0

            # Optional clamp: if you know memmap size, drop invalid ids
            if self.max_valid_his_id is not None and (v < 0 or v >= self.max_valid_his_id):
                v = 0

            safe_his_ids.append(v)
        his_id_list = safe_his_ids

        # Optional debug for the first few samples
        if idx < 3:
            print(f"[DEBUG] idx={idx} his_id_list (int): {his_id_list[:10]}")

        # ---------------------------------------------------------------------
        # [3A] Long-term profile history IDs (shape: max_his_len)
        # ---------------------------------------------------------------------
        his_id = self.pad_his(his_id_list, pad_to_len=self.max_his_len)

        # ---------------------------------------------------------------------
        # [3B] Short-term session IDs (recent slice of history)
        #   - derived from tail of his_id_list to avoid double-counting signal
        #   - shape: max_session_len
        # ---------------------------------------------------------------------
        recent_session_ids = his_id_list[-self.max_session_len:]
        session_ids = self.pad_his(recent_session_ids, pad_to_len=self.max_session_len)

        # ---------------------------------------------------------------------
        # [3C] Graph node IDs aligned with history
        #   - graph_node_ids_list aligns 1-to-1 with raw his_id_list
        #   - missing nodes set to -1
        #   - graph_node_mask is 1 if node exists else 0
        #   - both are padded to max_his_len to align with `his_id`
        # ---------------------------------------------------------------------
        graph_node_ids_list = []
        graph_node_mask_list = []
        for hid in his_id_list:
            norm_key = self.normalize_his_id(hid)

            if norm_key is None:
                node_id = -1
            else:
                node_id = self.his_to_graph.get(norm_key, -1)

            try:
                node_id = int(node_id)
            except (TypeError, ValueError):
                node_id = -1

            graph_node_ids_list.append(node_id)
            graph_node_mask_list.append(0 if node_id == -1 else 1)

        graph_node_ids = self.pad_his(graph_node_ids_list, pad_to_len=self.max_his_len)
        graph_node_mask = self.pad_his(graph_node_mask_list, pad_to_len=self.max_his_len)

        # ---------------------------------------------------------------------
        # [4A] Tokenize input for the LLM (Flan-T5)
        #   - padding='max_length' to keep tensors fixed
        #   - append special personalization tokens:
        #       [INST_PER_TOKEN], [SPC_PER_TOKEN]
        # ---------------------------------------------------------------------
        llm_encoded = self.llm_tokenizer(
            input_str,
            max_length=self.max_input_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        llm_input_ids = llm_encoded["input_ids"].squeeze(0)  # (seq,) on CPU

        # Append special personalization tokens (must exist in tokenizer vocab)
        inst_id = self.llm_tokenizer.convert_tokens_to_ids("[INST_PER_TOKEN]")
        spc_id = self.llm_tokenizer.convert_tokens_to_ids("[SPC_PER_TOKEN]")
        
        # ✅ CUDA FIX: Ensure special tokens created on same device (CPU)
        special_tokens = torch.tensor([inst_id, spc_id], dtype=torch.long, device=llm_input_ids.device)
        llm_input_ids = torch.cat([llm_input_ids, special_tokens])

        # Crop to max length after adding tokens
        if llm_input_ids.size(0) > self.max_input_len:
            llm_input_ids = llm_input_ids[:self.max_input_len]

        # NOTE: attention mask here is set to ones for the full (possibly padded) sequence.
        # That matches the original code behavior; if you want "true" padding masks,
        # you would instead take llm_encoded["attention_mask"] and extend it for appended tokens.
        llm_attention_mask = torch.ones_like(llm_input_ids)

        # ---------------------------------------------------------------------
        # [4B] Tokenize input for the embedding model (BGE)
        # ---------------------------------------------------------------------
        emb_encoded = self.emb_tokenizer(
            input_str,
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt"
        )
        emb_input_ids = emb_encoded["input_ids"].squeeze(0)  # (seq,) on CPU
        emb_attention_mask = emb_encoded["attention_mask"].squeeze(0)

        # Keep token_type_ids for compatibility (many models ignore it)
        # ✅ CUDA FIX: Create token_type_ids on same device as input_ids
        emb_token_type_ids = torch.zeros_like(emb_input_ids)

        # Crop embedding inputs too (defensive)
        if emb_input_ids.size(0) > self.max_input_len:
            emb_input_ids = emb_input_ids[:self.max_input_len]
            emb_attention_mask = emb_attention_mask[:self.max_input_len]
            emb_token_type_ids = emb_token_type_ids[:self.max_input_len]

        # ---------------------------------------------------------------------
        # [5] Tokenize output/label text
        # ---------------------------------------------------------------------
        labels = self.llm_tokenizer(
            output_str,
            max_length=self.max_new_len,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze(0)  # (tgt,) on CPU

        # ✅ CUDA COMPLIANCE: All tensors returned are on CPU
        # DataLoader/collator will:
        #   1. Batch these samples
        #   2. Move batch to device via .to(device)
        #   3. Pass to model.forward()
        return {
            "llm_input_ids": llm_input_ids,              # CPU tensor
            "llm_attention_mask": llm_attention_mask,    # CPU tensor
            "labels": labels,                            # CPU tensor
            "emb_input_ids": emb_input_ids,              # CPU tensor
            "emb_attention_mask": emb_attention_mask,    # CPU tensor
            "emb_token_type_ids": emb_token_type_ids,    # CPU tensor
            "his_id": his_id,                            # CPU tensor
            "session_ids": session_ids,                  # CPU tensor
            "graph_node_ids": graph_node_ids,            # CPU tensor
            "graph_node_mask": graph_node_mask           # CPU tensor
        }


    # =====================================================================
    # [PARSER] Read JSONL line → (input_str, output_str, his_id_list)
    # =====================================================================
    def parse_data(self, line):
        """
        Parse a JSONL line from the dataset file.

        Key behavior:
          - We truncate the *string* input by first tokenizing with llm_tokenizer,
            then decoding back. This prevents downstream warnings and keeps the
            input bounded for BOTH tokenizers.

        Returns:
            input_str: truncated string for model input
            output_str: label string
            his_id_list: raw list of history IDs
        """
        data = json.loads(line)

        # Tokenize once with truncation
        token_ids = self.llm_tokenizer.encode(
            data["input"],
            truncation=True,
            max_length=self.max_input_len
        )

        # Decode truncated ids back into text for downstream tokenizers
        input_str = self.llm_tokenizer.decode(token_ids, skip_special_tokens=True)

        output_str = data["output"]
        his_id_list = data["his_id"]

        return input_str, output_str, his_id_list