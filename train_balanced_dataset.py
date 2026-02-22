"""
Training script for the balanced 30-word ISL dataset.
Uses ResNet18 with transfer learning.
"""

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, random_split, Subset
from word_image_loader import WordImageDataset
from word_model import WordCNN

# -------------------------
# PATHS
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "data_Set")  # New balanced dataset
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "word_model_30classes.pth")

# -------------------------
# HYPERPARAMETERS (Optimized for 30 classes, 1500 images)
# -------------------------
BATCH_SIZE = 32          # Larger batch for better gradient estimates
LEARNING_RATE = 0.0001   # Good starting LR for fine-tuning
WEIGHT_DECAY = 0.01      # L2 regularization
EPOCHS = 50              # Train for all 50 epochs
EARLY_STOPPING = False   # Disabled - train all epochs
PATIENCE = 10            # (Not used when EARLY_STOPPING=False)
VAL_SPLIT = 0.2          # 20% validation

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# DATASET LOADING
# -------------------------
print(f"\nLoading dataset from: {DATASET_PATH}")

# Create base dataset (without augmentation for splitting)
base_dataset = WordImageDataset(DATASET_PATH, augment=False)
num_classes = len(base_dataset.labels)
print(f"Number of classes: {num_classes}")
print(f"Total samples: {len(base_dataset)}")
print(f"Classes: {base_dataset.labels}")

# Split indices
total_size = len(base_dataset)
val_size = int(VAL_SPLIT * total_size)
train_size = total_size - val_size

# Random split indices
indices = torch.randperm(total_size).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create separate datasets with proper augmentation settings
train_dataset_aug = WordImageDataset(DATASET_PATH, augment=True)
val_dataset_noaug = WordImageDataset(DATASET_PATH, augment=False)

# Create subsets
train_dataset = Subset(train_dataset_aug, train_indices)
val_dataset = Subset(val_dataset_noaug, val_indices)

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# -------------------------
# MODEL
# -------------------------
model = WordCNN(num_classes).to(device)
print(f"\nModel: ResNet18 with {num_classes} output classes")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# -------------------------
# OPTIMIZER + LOSS + SCHEDULER
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# TRAINING LOOP WITH EARLY STOPPING
# -------------------------
print("\n" + "="*60)
print("Starting Training")
print("="*60)

best_val_acc = 0.0
epochs_without_improvement = 0
training_history = []

for epoch in range(EPOCHS):
    # ----- Training Phase -----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total

    # ----- Validation Phase -----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    # Update scheduler
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']

    # Save history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'train_acc': train_acc,
        'val_loss': avg_val_loss,
        'val_acc': val_acc,
        'lr': current_lr
    })

    # Print progress
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:5.1f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:5.1f}% | "
          f"LR: {current_lr:.6f}")

    # Check for improvement and save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'num_classes': num_classes,
            'class_labels': base_dataset.labels
        }, MODEL_SAVE_PATH)
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.1f}%)")
    else:
        epochs_without_improvement += 1
        # Early stopping (only if enabled)
        if EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"\n⚠ Early stopping triggered after {PATIENCE} epochs without improvement")
            break

# -------------------------
# TRAINING SUMMARY
# -------------------------
print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")

# Final recommendations
print("\n" + "-"*60)
print("Recommendations:")
print("-"*60)
if best_val_acc >= 90:
    print("✓ Excellent accuracy! Model is ready for deployment.")
elif best_val_acc >= 80:
    print("✓ Good accuracy. Consider collecting more training data for improvement.")
elif best_val_acc >= 70:
    print("⚠ Moderate accuracy. Try:")
    print("  - Increasing augmentation diversity")
    print("  - Using a larger model (ResNet34)")
    print("  - Collecting more data")
else:
    print("⚠ Low accuracy. Consider:")
    print("  - Checking data quality and labels")
    print("  - More aggressive data augmentation")
    print("  - Using a different model architecture")

