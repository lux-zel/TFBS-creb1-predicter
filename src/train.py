# train.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from config import *
from model import BindingCNN
import os

#data laden
positive = np.load(os.path.join(DATA_DIR, "positive_seqs.npy"))
negative = np.load(os.path.join(DATA_DIR, "negative_seqs.npy"))

X = np.concatenate([positive, negative])
y = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# naar torch tensors veranderen
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model, loss, optimizer
device = torch.device(DEVICE)
model = BindingCNN(seq_len=SEQ_LEN).to(device)
# weighted BCE loss voor recurance problemen
pos_weight = (len(y_train[y_train==0]) / len(y_train[y_train==1]))
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validatie
    model.eval()
    val_loss = 0
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            preds.extend(outputs.cpu().numpy())
            true.extend(yb.cpu().numpy())
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_auc = roc_auc_score(true, preds)
    val_auprc = average_precision_score(true, preds)
    print(f"Epoch {epoch+1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, AUROC {val_auc:.4f}, AUPRC {val_auprc:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_model.pt"))

# Test evaluation
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_model.pt")))
model.eval()
preds, true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb).cpu().numpy()
        preds.extend(outputs)
        true.extend(yb.numpy())
test_auc = roc_auc_score(true, preds)
test_auprc = average_precision_score(true, preds)
print(f"\nTest AUROC: {test_auc:.4f}, Test AUPRC: {test_auprc:.4f}")