
#!/bin/bash
    # --subsample-train 5000 \
    # --subsample-eval 500 \
python train_with_optuna_fast.py \
    --llm-model-path ../FlanT5-Large \
    --emb-model-path ../bge-base-en-v1.5 \
    --train-file ../LaMP_time_3_subset_id/train_aug_input.json \
    --eval-file ../LaMP_time_3_subset_id/dev_profile.json \
    --graph-dir ../graph_emb \
    --bge-emb-dir ../bge_emb \
    --task-id 3 \
    --max-input-len 256 \
    --max-new-len 10 \
    --max-his-len 10 \
    --max-session-len 5 \
    \
    --num-epochs 2 \
    --batch-size 16 \
    --logging-steps 100 \
    \
    --n-trials 20 \
    --use-profile true \
    --use-session true \
    --use-graph true \
    \
    --optuna-output-dir ./optuna_trials_3k \
    --study-name stageab_3k_2ep_hpo \
    --storage sqlite:///optuna_3k.db \
    2>&1 | tee optuna_3k.log