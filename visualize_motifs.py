# visualize_motifs.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logomaker

#importeer het model structuur
from model import BindingCNN
from config import DATA_DIR, SEQ_LEN, DEVICE

#laad het getrainde model
model_path = os.path.join(DATA_DIR, "best_model.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

model = BindingCNN(seq_len=SEQ_LEN)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print("Model loaded successfully.")

#de laag filters krijgen
filters = model.conv1.weight.data.cpu().numpy()
n_filters = filters.shape[0]         
kernel_size = filters.shape[2]         
print(f"First layer has {n_filters} filters of length {kernel_size}.")

#maak een plot met 16 filters
n_plot = min(16, n_filters)
rows = 4
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()

for i in range(n_plot):
   
    pwm = filters[i]
  
    pwm = pwm - pwm.mean(axis=0, keepdims=True)

    pwm = np.exp(pwm) / np.exp(pwm).sum(axis=0, keepdims=True)

    pwm_df = pd.DataFrame(pwm.T, columns=['A', 'C', 'G', 'T'])
    
 
    logo = logomaker.Logo(pwm_df, ax=axes[i], font_name='Arial')
    axes[i].set_ylabel(f'Filter {i+1}')
    axes[i].set_ylim(0, 1)


for j in range(n_plot, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "learned_motifs.png"), dpi=150)
plt.show()
print(f"Motif plot saved to {os.path.join(DATA_DIR, 'learned_motifs.png')}")