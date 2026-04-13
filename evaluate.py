import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

from dataset import get_loaders
from model import CustomCNN, get_resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "./checkpoints"


@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def plot_curves(model_name):
    history = torch.load(f"{CKPT_DIR}/history_{model_name}.pth")
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_acc"], label="Train")
    ax1.plot(epochs, history["val_acc"],   label="Val")
    ax1.set_title(f"{model_name.upper()} — Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history["train_loss"], label="Train")
    ax2.plot(epochs, history["val_loss"],   label="Val")
    ax2.set_title(f"{model_name.upper()} — Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{CKPT_DIR}/curves_{model_name}.png", dpi=150)
    plt.show()
    print(f"Saved curves to {CKPT_DIR}/curves_{model_name}.png")


def evaluate(model_name="cnn"):
    _, _, test_loader = get_loaders()

    model = CustomCNN() if model_name == "cnn" else get_resnet18(freeze_backbone=False)
    model.load_state_dict(torch.load(f"{CKPT_DIR}/best_{model_name}.pth", map_location=DEVICE))
    model = model.to(DEVICE)

    preds, labels = get_preds(model, test_loader)
    acc = (preds == labels).mean()
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(classification_report(labels, preds, digits=4))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=90)
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")
    plt.tight_layout()
    plt.savefig(f"{CKPT_DIR}/confusion_{model_name}.png", dpi=150)
    plt.show()
    print(f"Saved confusion matrix to {CKPT_DIR}/confusion_{model_name}.png")

    plot_curves(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    args = parser.parse_args()
    evaluate(args.model)
