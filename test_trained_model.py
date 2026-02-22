"""
Test the trained ISL model on the validation set.
Run this after training to see detailed accuracy metrics.
"""

import torch
import os
from torch.utils.data import DataLoader, Subset
from word_image_loader import WordImageDataset
from word_model import WordCNN
from collections import defaultdict

# -------------------------
# PATHS
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "data_Set")
MODEL_PATH = os.path.join(PROJECT_DIR, "word_model_30classes.pth")

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# LOAD MODEL
# -------------------------
print(f"\nLoading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found! Please train the model first.")
    print("Run: python3 scripts/train_balanced_dataset.py")
    exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = checkpoint['num_classes']
class_labels = checkpoint['class_labels']

print(f"Model trained for {num_classes} classes")
print(f"Best validation accuracy during training: {checkpoint['val_acc']:.2f}%")

model = WordCNN(num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# -------------------------
# LOAD TEST DATA
# -------------------------
print(f"\nLoading test data from: {DATASET_PATH}")
dataset = WordImageDataset(DATASET_PATH, augment=False)

# Use 20% as test set (same split ratio as training)
total_size = len(dataset)
test_size = int(0.2 * total_size)
train_size = total_size - test_size

# Use same random seed for reproducible split
torch.manual_seed(42)
indices = torch.randperm(total_size).tolist()
test_indices = indices[train_size:]

test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Test samples: {len(test_dataset)}")

# -------------------------
# EVALUATE
# -------------------------
print("\n" + "="*60)
print("Evaluating Model")
print("="*60)

correct = 0
total = 0
class_correct = defaultdict(int)
class_total = defaultdict(int)
predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            class_name = class_labels[label]
            class_total[class_name] += 1
            if label == pred:
                class_correct[class_name] += 1
            predictions.append((class_name, class_labels[pred], label == pred))

# -------------------------
# RESULTS
# -------------------------
overall_acc = 100.0 * correct / total

print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {overall_acc:.2f}%")
print(f"{'='*60}")

print(f"\n{'Class':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
print("-" * 55)

for class_name in sorted(class_labels):
    if class_total[class_name] > 0:
        acc = 100.0 * class_correct[class_name] / class_total[class_name]
        print(f"{class_name:<20} {acc:>9.1f}% {class_correct[class_name]:>10} {class_total[class_name]:>10}")

# Find worst performing classes
print(f"\n{'='*60}")
print("Classes that need improvement (accuracy < 80%):")
print("="*60)

low_acc_classes = []
for class_name in class_labels:
    if class_total[class_name] > 0:
        acc = 100.0 * class_correct[class_name] / class_total[class_name]
        if acc < 80:
            low_acc_classes.append((class_name, acc))

if low_acc_classes:
    for class_name, acc in sorted(low_acc_classes, key=lambda x: x[1]):
        print(f"  {class_name}: {acc:.1f}%")
else:
    print("  All classes have accuracy >= 80%! Great job!")

print(f"\n{'='*60}")
print("Model is ready for real-time detection!")
print("Run: python3 scripts/realtime_30words.py")
print("="*60)

