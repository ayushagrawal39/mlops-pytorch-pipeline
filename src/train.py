"""Train the CNN; every stdout record is JSON for easy log ingestion."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from dataset import get_dataloaders
from model import get_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, loss_fn, optimizer, device):
    model.train() if optimizer else model.eval()
    total_loss = correct = total = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer:
                optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.getenv("TRAINING_CONFIG", "configs/training_config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, v = cfg["training"], cfg["data"]
    train_loader, val_loader = get_dataloaders(v["data_dir"], t["batch_size"], t.get("num_workers", 0))
    model = get_model(cfg["model"]["architecture"], cfg["model"]["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    checkpoint = Path(cfg["output"]["checkpoint_dir"]) / cfg["output"]["model_name"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, stale = float("inf"), 0
    for epoch in range(1, t["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, None, device)
        print(json.dumps({"epoch": epoch, "train_loss": round(train_loss, 5),
                          "train_accuracy": round(train_acc, 5), "val_loss": round(val_loss, 5),
                          "val_accuracy": round(val_acc, 5)}), flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "num_classes": cfg["model"]["num_classes"],
                        "architecture": cfg["model"]["architecture"], "val_loss": val_loss}, checkpoint)
            print(json.dumps({"event": "checkpoint_saved", "path": str(checkpoint)}), flush=True)
        else:
            stale += 1
            if stale >= t["early_stopping_patience"]:
                print(json.dumps({"event": "early_stopping", "epoch": epoch}), flush=True)
                break
    print(json.dumps({"event": "training_complete", "best_val_loss": round(best, 5)}), flush=True)


if __name__ == "__main__":
    main()
