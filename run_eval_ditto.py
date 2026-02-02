import argparse
import json
import subprocess
import time
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def read_true_labels(test_path):
    y_true = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            y_true.append(int(line.rstrip("\n").split("\t")[-1]))
    return y_true


def read_pred_labels(pred_path):
    y_pred = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            # molte implementazioni scrivono "0" o "1" per riga
            y_pred.append(int(line.strip()))
    return y_pred


def main():
    p = argparse.ArgumentParser(description="Train + Predict Ditto + metriche + tempi (FAIR-DA4ER)")
    p.add_argument("--task", required=True, help="Nome task (es: cars_B1)")
    p.add_argument("--train_path", required=True)
    p.add_argument("--valid_path", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--pred_out", required=True, help="File predizioni (una label per riga)")
    p.add_argument("--metrics_out", required=True, help="JSON metriche")
    args = p.parse_args()

    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)

    # 1) TRAIN
    t_train_start = time.perf_counter()
    subprocess.check_call([
        "python", "train.py",
        "--task", args.task,
        "--train_path", args.train_path,
        "--valid_path", args.valid_path
    ])
    t_train_end = time.perf_counter()

    # 2) PREDICT
    t_inf_start = time.perf_counter()
    subprocess.check_call([
        "python", "predict.py",
        "--task", args.task,
        "--test_path", args.test_path,
        "--output_path", str(pred_out)
    ])
    t_inf_end = time.perf_counter()

    # 3) EVAL
    y_true = read_true_labels(args.test_path)
    y_pred = read_pred_labels(pred_out)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "training_time": float(t_train_end - t_train_start),
        "inference_time": float(t_inf_end - t_inf_start),
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Metriche Ditto salvate in:", args.metrics_out)


if __name__ == "__main__":
    main()
