
import json
import os
import random
from collections import defaultdict, Counter

BASE_DIR = ".."
TASK_ID = 3
SUBSET_SIZE_TRAIN = 500   # desired total samples
SUBSET_SIZE_DEV = 200     # desired total samples
SEED = 42

MAX_PROFILE_ITEMS = 50  # only keep first N history entries

# 🆕 Trim profile history size for each question
# Option A (default): randomly keep between 8 and 15 history items
# Option B: keep full history (no trimming)
USE_FULL_HISTORY = False

random.seed(SEED)

lamp_dir = os.path.join(BASE_DIR, f"LaMP_time_{TASK_ID}")
subset_dir = os.path.join(BASE_DIR, f"LaMP_time_{TASK_ID}_subset")
os.makedirs(subset_dir, exist_ok=True)

def stratified_subset_pair(q_file, o_file, dst_q_file, dst_o_file, subset_size):
    """
    Create a subset with equal class distribution based on outputs['golds'][i]['output'].
    If a class does not have enough samples, oversample with replacement.
    Output order matches question order exactly for aggr_id compatibility.
    """
    questions = json.load(open(q_file))
    outputs = json.load(open(o_file))

    # Build ID → output mapping
    id_to_output = {o["id"]: o for o in outputs["golds"]}
    id_to_label = {o["id"]: str(o["output"]) for o in outputs["golds"]}

    # Group question entries by class label
    class_to_questions = defaultdict(list)
    for q in questions:
        label = id_to_label.get(q["id"])
        if label is not None:
            class_to_questions[label].append(q)

    classes = sorted(class_to_questions.keys())
    num_classes = len(classes)

    # How many samples per class
    per_class = subset_size // num_classes
    remainder = subset_size % num_classes

    subset_questions = []
    for i, cls in enumerate(classes):
        candidates = class_to_questions[cls]
        take_n = per_class + (1 if i < remainder else 0)  # spread remainder

        if len(candidates) < take_n:
            print(f"⚠️ Class {cls}: only {len(candidates)} samples, oversampling to {take_n}")
            extras_needed = take_n - len(candidates)
            extras = random.choices(candidates, k=extras_needed)  # duplicates allowed
            subset_questions.extend(candidates + extras)
        else:
            subset_questions.extend(random.sample(candidates, take_n))

    # # 🆕 Trim profile history size for each question
    # for q in subset_questions:
    #     if "profile" in q and isinstance(q["profile"], list):
    #         q["profile"] = q["profile"][:MAX_PROFILE_ITEMS]


    for q in subset_questions:
        if "profile" in q and isinstance(q["profile"], list):
            if USE_FULL_HISTORY:
                continue  # keep all history for this id
            # Pick a random length between 8 and 15
            max_items = random.randint(15, 40)
            q["profile"] = q["profile"][:max_items]

    # Shuffle to avoid class order bias
    random.shuffle(subset_questions)

    # Maintain exact question-output alignment
    subset_ids_order = [q["id"] for q in subset_questions]
    outputs_subset_ordered = {
        "golds": [id_to_output[q_id] for q_id in subset_ids_order]
    }

    # Save files
    json.dump(subset_questions, open(dst_q_file, "w"), ensure_ascii=False, indent=2)
    json.dump(outputs_subset_ordered, open(dst_o_file, "w"), ensure_ascii=False, indent=2)

    print(f"✅ Balanced {os.path.basename(dst_q_file)} aligned with {os.path.basename(dst_o_file)}")
    counts = Counter(id_to_label[i] for i in subset_ids_order)
    print(f"   🎯 Final class distribution: {dict(counts)} (Total: {len(subset_questions)})")


# Train balanced subset
stratified_subset_pair(
    os.path.join(lamp_dir, "train_questions.json"),
    os.path.join(lamp_dir, "train_outputs.json"),
    os.path.join(subset_dir, "train_questions.json"),
    os.path.join(subset_dir, "train_outputs.json"),
    SUBSET_SIZE_TRAIN
)

# Dev balanced subset
stratified_subset_pair(
    os.path.join(lamp_dir, "dev_questions.json"),
    os.path.join(lamp_dir, "dev_outputs.json"),
    os.path.join(subset_dir, "dev_questions.json"),
    os.path.join(subset_dir, "dev_outputs.json"),
    SUBSET_SIZE_DEV
)

print("✅ All balanced subset files regenerated in", subset_dir)
