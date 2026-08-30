"""Test-stage error analysis for a trained real-vs-AI image detector.

The script uses only labelled test folders. It never trains or changes a model.
By default it analyses clean images; set ERROR_TRANSFORMS=all to repeat the
analysis for every required corruption.

Run:
    python error_analysis.py
    MODEL_PATH=model_v3.pth python error_analysis.py
    MODEL_PATH=model_v3.pth ERROR_TRANSFORMS=all python error_analysis.py
"""
import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config
from data import ImageFolderDataset, eval_transform, get_corruptions
from model import load_model


OUT_ROOT = Path(os.environ.get("ERROR_OUT", "error_analysis"))
MAX_EXAMPLES = int(os.environ.get("ERROR_EXAMPLES", 25))


def outcome_name(true_label, predicted_label):
    """Positive means AI-generated (label 1); negative means real (label 0)."""
    if true_label == 1 and predicted_label == 1:
        return "true_positive"
    if true_label == 0 and predicted_label == 0:
        return "true_negative"
    if true_label == 0 and predicted_label == 1:
        return "false_positive"
    return "false_negative"


def calculate_metrics(rows):
    counts = {name: 0 for name in (
        "true_positive", "true_negative", "false_positive", "false_negative"
    )}
    for row in rows:
        counts[row["outcome"]] += 1

    tp, tn = counts["true_positive"], counts["true_negative"]
    fp, fn = counts["false_positive"], counts["false_negative"]
    total = tp + tn + fp + fn

    def divide(a, b):
        return a / b if b else 0.0

    return {
        **counts,
        "total": total,
        "accuracy": divide(tp + tn, total),
        "precision_ai": divide(tp, tp + fp),
        "recall_ai": divide(tp, tp + fn),
        "f1_ai": divide(2 * tp, 2 * tp + fp + fn),
        "specificity_real": divide(tn, tn + fp),
    }


def analyze_transform(model, device, name, corruption):
    dataset = ImageFolderDataset(config.TEST_DIRS, eval_transform(corruption))
    if not dataset.samples:
        raise RuntimeError("No labelled test images were found under config.TEST_DIRS")

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.get_num_workers(),
        pin_memory=config.use_pin_memory(),
    )
    rows = []
    sample_index = 0
    with torch.no_grad():
        for images, true_labels in loader:
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)[:, 1].cpu()
            predicted_labels = (probabilities >= config.THRESHOLD).long()

            for true_label, predicted_label, ai_score in zip(
                true_labels.tolist(), predicted_labels.tolist(), probabilities.tolist()
            ):
                path = dataset.samples[sample_index][0]
                sample_index += 1
                rows.append({
                    "transformation": name,
                    "image_path": path,
                    "true_label": config.CLASS_NAMES[true_label],
                    "predicted_label": config.CLASS_NAMES[predicted_label],
                    "ai_score": ai_score,
                    "outcome": outcome_name(true_label, predicted_label),
                })
    return rows


def save_confusion_matrix(metrics, path):
    # Rows are ground truth; columns are predictions.
    matrix = [
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Real", "AI-generated"],
        yticklabels=["Real", "AI-generated"],
        xlabel="Predicted label", ylabel="Correct label",
        title="Confusion matrix (clean test images)",
    )
    for row in range(2):
        for column in range(2):
            ax.text(column, row, matrix[row][column], ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_error_examples(rows, output_dir):
    """Save the most confidently wrong clean-test examples for inspection."""
    errors = [row for row in rows if row["outcome"] in ("false_positive", "false_negative")]
    errors.sort(
        key=lambda row: row["ai_score"]
        if row["outcome"] == "false_positive" else 1 - row["ai_score"],
        reverse=True,
    )
    saved = {"false_positive": 0, "false_negative": 0}
    for row in errors:
        outcome = row["outcome"]
        if saved[outcome] >= MAX_EXAMPLES:
            continue
        source = Path(row["image_path"])
        destination_dir = output_dir / outcome
        destination_dir.mkdir(parents=True, exist_ok=True)
        score = round(100 * row["ai_score"])
        destination = destination_dir / f"{saved[outcome]:03d}_ai-{score}_{source.name}"
        shutil.copy2(source, destination)
        saved[outcome] += 1
    return saved


def selected_corruptions():
    corruptions = get_corruptions()
    selection = os.environ.get("ERROR_TRANSFORMS", "clean").strip().lower()
    if selection == "all":
        return corruptions
    names = [name.strip() for name in selection.split(",") if name.strip()]
    unknown = [name for name in names if name not in corruptions]
    if unknown:
        raise ValueError(f"Unknown transformations: {unknown}. Choose from {list(corruptions)}")
    return {name: corruptions[name] for name in names}


def main():
    device = config.get_device()
    model = load_model(device)
    model_name = Path(config.MODEL_PATH).stem
    output_dir = OUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {
        "model_path": config.MODEL_PATH,
        "threshold": config.THRESHOLD,
        "test_directories": config.TEST_DIRS,
        "transformations": {},
    }

    print(f"[error analysis] model={config.MODEL_PATH} device={device}")
    for name, corruption in selected_corruptions().items():
        rows = analyze_transform(model, device, name, corruption)
        metrics = calculate_metrics(rows)
        summary["transformations"][name] = metrics
        all_rows.extend(rows)
        print(
            f"{name:<16} accuracy={metrics['accuracy']:.2%} "
            f"precision={metrics['precision_ai']:.2%} recall={metrics['recall_ai']:.2%} "
            f"TP={metrics['true_positive']} TN={metrics['true_negative']} "
            f"FP={metrics['false_positive']} FN={metrics['false_negative']}"
        )

        if name == "clean":
            save_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
            summary["saved_error_examples"] = save_error_examples(rows, output_dir / "examples")

    fieldnames = [
        "transformation", "image_path", "true_label", "predicted_label",
        "ai_score", "outcome",
    ]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"[error analysis] reports saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
