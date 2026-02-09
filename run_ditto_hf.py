import os, time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

MODEL_NAME = "distilroberta-base"   # più leggero di roberta-base, ma sempre “ditto-like”
MAX_LEN    = 256                   # puoi alzare se hai GPU buona
BATCH_TRAIN = 16                   # adatta alla tua GPU/CPU
BATCH_EVAL  = 64

DATA_DIR = "ditto_data"
OUT_DIR  = "ditto_models"

def read_ditto_file(path: str) -> List[Tuple[str, str, int]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a, b, y = line.rstrip("\n").split("\t")
            data.append((a, b, int(y)))
    return data

class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tok = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, y = self.pairs[idx]
        enc = self.tok(
            a, b,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item

def best_threshold_f1(y_true, y_score) -> float:
    p, r, thr = precision_recall_curve(y_true, y_score)
    f1 = 2*p*r/(p+r+1e-12)
    # thr ha len = len(p)-1
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])

def eval_at(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

@torch.no_grad()
def predict_scores(model, tokenizer, pairs, batch_size=BATCH_EVAL):
    model.eval()
    y_true = np.array([y for _, _, y in pairs], dtype=int)
    scores = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        a = [x[0] for x in batch]
        b = [x[1] for x in batch]
        enc = tokenizer(a, b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        scores.append(prob)

    return y_true, np.concatenate(scores, axis=0)

def train_once():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_pairs = read_ditto_file(os.path.join(DATA_DIR, "base", "train.txt"))
    valid_pairs = read_ditto_file(os.path.join(DATA_DIR, "base", "valid.txt"))

    train_ds = PairDataset(train_pairs, tokenizer)
    valid_ds = PairDataset(valid_pairs, tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "base"),
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        num_train_epochs=1,               # aumenta a 2-3 se hai GPU e tempo
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=200,
        eval_strategy="no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    t0 = time.time()
    trainer.train()
    t1 = time.time()
    train_time = t1 - t0

    return model, tokenizer, train_time

def run_pipeline(strategy: str, model, tokenizer):
    assert strategy in ["B1", "B2"]
    val_path  = os.path.join(DATA_DIR, strategy, "valid_blocked.txt")
    test_path = os.path.join(DATA_DIR, strategy, "test_blocked.txt")

    val_pairs  = read_ditto_file(val_path)
    test_pairs = read_ditto_file(test_path)

    # VALID inference + threshold tuning
    t0 = time.time()
    yv, sv = predict_scores(model, tokenizer, val_pairs)
    t1 = time.time()
    thr = best_threshold_f1(yv, sv)
    val_m = eval_at(yv, sv, thr)
    val_inf = t1 - t0

    # TEST inference
    t2 = time.time()
    yt, st = predict_scores(model, tokenizer, test_pairs)
    t3 = time.time()
    test_m = eval_at(yt, st, thr)
    test_inf = t3 - t2

    print(f"[{strategy}] VALID - pairs={len(yv)} thr={thr:.4f} "
          f"P={val_m['precision']:.4f} R={val_m['recall']:.4f} F1={val_m['f1']:.4f} "
          f"(inference_time={val_inf:.2f}s)")

    print(f"[{strategy}] TEST  - pairs={len(yt)} thr={thr:.4f} "
          f"P={test_m['precision']:.4f} R={test_m['recall']:.4f} F1={test_m['f1']:.4f} "
          f"(inference_time={test_inf:.2f}s)")

    return {
        "pipeline": f"{strategy}-ditto",
        "thr": thr,
        "val": {**val_m, "pairs": int(len(yv)), "infer_s": float(val_inf)},
        "test": {**test_m, "pairs": int(len(yt)), "infer_s": float(test_inf)},
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t_all0 = time.time()
    model, tokenizer, train_time = train_once()
    train_time = float(train_time)

    # sposta su GPU se c’è
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    r1 = run_pipeline("B1", model, tokenizer)
    r2 = run_pipeline("B2", model, tokenizer)
    t_all1 = time.time()

    print("\n" + "="*80)
    print("REPORT FINALE (DITTO-style, NO SAMPLING)")
    print("="*80)
    print(f"Training_time_s: {train_time:.2f}s")
    print(f"Total_time_s: {t_all1 - t_all0:.2f}s\n")
    for r in [r1, r2]:
        print(f"{r['pipeline']:10s} Thr={r['thr']:.4f} "
              f"ValF1={r['val']['f1']:.4f} TestF1={r['test']['f1']:.4f} "
              f"TestP={r['test']['precision']:.4f} TestR={r['test']['recall']:.4f} "
              f"TestPairs={r['test']['pairs']} TestInferS={r['test']['infer_s']:.2f}s")

if __name__ == "__main__":
    main()
