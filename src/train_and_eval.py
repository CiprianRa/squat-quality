import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_bilstm_attention import BiLSTMWithAttention
from skeleton_sequence import SkeletonSequenceDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import Counter
import json


# Config
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "../data/"
# Load datasets
train_dataset = SkeletonSequenceDataset(ROOT_DIR, "../data/train_files.txt")
val_dataset = SkeletonSequenceDataset(ROOT_DIR, "../data/val_files.txt", label_map=train_dataset.label_to_idx)
test_dataset = SkeletonSequenceDataset(ROOT_DIR, "../data/test_files.txt", label_map=train_dataset.label_to_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Compute class weights
label_counts = Counter(train_dataset.labels)
total = sum(label_counts.values())
weights = torch.tensor(
    [total / label_counts[cls] for cls in sorted(train_dataset.label_to_idx.keys())],
    dtype=torch.float32
)
weights = weights / weights.sum()  # Normalize
weights = weights.to(DEVICE)

# Model
model = BiLSTMWithAttention(input_dim=28, hidden_dim=128, output_dim=len(train_dataset.label_to_idx), num_layers=1, dropout=0.3)
model.to(DEVICE)

# Loss & optimizer
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0.0
best_model_state = None

print(f"\nEtichete detectate: {train_dataset.label_to_idx}")
print("\n===== Încep antrenarea =====")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(y_true, y_pred, average="macro")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_state = model.state_dict()

    print(f"Epoca {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

# ===== Test final =====
print("\n===== Evaluare finală pe TEST =====")
model.load_state_dict(best_model_state)
model.eval()
test_loss = 0
correct = 0
total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        outputs = model(X)
        loss = criterion(outputs, y)
        test_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_acc = 100 * correct / total
print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

# === Salvare model antrenat ===
torch.save(model.state_dict(),"../models/model_bilstm_attention.pt")
print("Model salvat în 'model_bilstm_attention.pt'")

# === Salvare dictionar etichete → index ===
with open("../models/label_to_idx.json", "w") as f:
    json.dump(train_dataset.label_to_idx, f, indent=2)
print("Etichete salvate în 'label_to_idx.json'")

# Confusion matrix
cm = confusion_matrix(all_targets, all_preds)
labels = list(test_loader.dataset.label_to_idx.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("../confusion_matrix.png")
print("Confusion matrix salvata ca 'confusion_matrix.png'")

# Classification report
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=labels, digits=4))
