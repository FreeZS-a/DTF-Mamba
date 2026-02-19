import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.model import SCMNet
from configs.config_LoveDA import dataset as dataset_cfg, train as train_cfg
from data_loader import get_dataset


def compute_miou(pred, target, num_classes):
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = get_dataset(dataset_cfg)
    val_dataset = get_dataset({**dataset_cfg, 'split': 'val'})

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SCMNet(num_classes=dataset_cfg['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'])

    best_miou = 0.0
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(train_cfg['epochs']):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        model.eval()
        val_miou = 0.0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                miou = compute_miou(outputs, labels, dataset_cfg['num_classes'])
                if not (isinstance(miou, float) and np.isnan(miou)):
                    val_miou += miou
                    val_count += 1
        val_miou = val_miou / val_count if val_count > 0 else 0.0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"Epoch {epoch+1}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Val mIoU = {val_miou:.4f}, "
              f"LR = {current_lr:.2e}")

        save_path = os.path.join(save_dir, "best_model.pth")
        low_loss_path = os.path.join(save_dir, "low_loss_model.pth")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with mIoU = {best_miou:.4f}")

        if train_loss < 1.0 and not os.path.exists(low_loss_path):
            torch.save(model.state_dict(), low_loss_path)
            print(f"💾 Low-loss model saved (loss={train_loss:.4f})")

    print(f"🎉 Training finished! Best Val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()