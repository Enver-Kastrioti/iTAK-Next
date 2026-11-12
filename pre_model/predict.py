#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Hot Encoding Multi-scale CNN Protein Prediction Script
- Multi-scale OneHotLightweightCNN model based on train_onehot_cnn.py
- Supports transcription factor prediction for single FASTA files
- Outputs protein prediction probability table
- Supports custom confidence threshold
- Adapted for train_data8.28 multi-scale CNN architecture
"""

import os
import sys
import argparse
import concurrent.futures
import hashlib

# Solve MKL library conflict - must be set before importing torch
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure fonts for CJK glyph rendering when available
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



# ---------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer_name='deep_conv2'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        # Locate the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
                
        if target_layer is None:
            # Fallback: handle DataParallel modules (e.g., module.deep_conv2)
            for name, module in self.model.named_modules():
                if name.endswith(self.target_layer_name):
                    target_layer = module
                    break
        
        if target_layer is None:
            print(f"Warning: Layer {self.target_layer_name} not found in model. Grad-CAM will not work.")
            return
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        
    def generate(self, input_tensor, target_class=None):
        # input_tensor: (1, max_len, vocab_size)
        # Note: we do not need to call model.eval() here because inference already uses eval mode.
        # However, we still need to zero gradients for the backward pass.
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            return None, output, target_class

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global Average Pooling of gradients to get weights
        weights = np.mean(gradients, axis=1) # (128,)
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :]
            
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        return cam, output, target_class

def plot_gradcam(sequence, cam, title, save_path):
    # cam is 1D array of length 1000 (or model max_length)
    # sequence is string
    
    # Ensure using non-interactive backend for thread/process safety
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Trim CAM to sequence length (if seq < max_length)
    seq_len = len(sequence)
    # The cam length corresponds to the model's max_length (usually 1000)
    # We should take the first seq_len elements if the input was padded
    # But wait, if the sequence was a fragment of length 1000, seq_len is 1000.
    
    cam_len = len(cam)
    if seq_len < cam_len:
        cam_trimmed = cam[:seq_len]
    else:
        cam_trimmed = cam
        seq_len = cam_len # Visualize only up to cam length

    cam_trimmed = np.asarray(cam_trimmed, dtype=np.float32)
    cam_max = float(np.max(cam_trimmed)) if cam_trimmed.size else 0.0
    if cam_max > 0:
        cam_trimmed = cam_trimmed / cam_max
    
    plt.figure(figsize=(15, 6))
    
    # Plot heatmap
    # Expand dims to (1, seq_len) for imshow
    plt.imshow(
        cam_trimmed[np.newaxis, :],
        aspect='auto',
        cmap='jet',
        alpha=0.6,
        extent=[0, seq_len, 0, 1],
        vmin=0.0,
        vmax=1.0,
    )
    
    # Plot sequence characters
    plt.plot(cam_trimmed, color='black', linewidth=1, alpha=0.5)
    plt.fill_between(range(seq_len), cam_trimmed, alpha=0.2, color='blue')
    
    cbar = plt.colorbar()
    cbar.set_label('Importance', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.title(title, fontsize=18)
    plt.xlabel('Sequence Position', fontsize=14)
    plt.ylabel('Activation', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add high activation residues text could be added here
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Amino acid to number mapping
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
aa_to_idx['X'] = len(AMINO_ACIDS)  # Unknown amino acid

class OneHotLightweightCNN(nn.Module):
    """Multi-scale CNN model using One-Hot encoding
    
    Uses three different configurations of multi-layer convolution kernel combinations to achieve multi-scale feature extraction through parallel structures:
    Config 1: (4x21, 4x1, 16x1)
    Config 2: (12x21, 8x1, 4x1) 
    Config 3: (16x21, 4x1, 4x1)
    """
    def __init__(self, vocab_size=21, num_classes=2, max_length=1000):
        super(OneHotLightweightCNN, self).__init__()
        
        # Input: (batch_size, max_length, vocab_size)
        # Needs to be transposed to: (batch_size, vocab_size, max_length) for Conv1d
        
        # Config 1: (4x21, 4x1, 16x1) - Small scale fine features
        self.branch1_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=4, padding=2)  # 4x21 equivalent
        self.branch1_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)        # 4x1
        self.branch1_conv3 = nn.Conv1d(128, 256, kernel_size=16, padding=8)      # 16x1
        
        # Config 2: (12x21, 8x1, 4x1) - Medium scale features
        self.branch2_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=12, padding=6) # 12x21 equivalent
        self.branch2_conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)        # 8x1
        self.branch2_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)       # 4x1
        
        # Config 3: (16x21, 4x1, 4x1) - Large scale global features
        self.branch3_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=16, padding=8) # 16x21 equivalent
        self.branch3_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)        # 4x1
        self.branch3_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)       # 4x1
        
        # Batch Normalization layers - Branch 1
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn1_3 = nn.BatchNorm1d(256)
        
        # Batch Normalization layers - Branch 2
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(256)
        
        # Batch Normalization layers - Branch 3
        self.bn3_1 = nn.BatchNorm1d(64)
        self.bn3_2 = nn.BatchNorm1d(128)
        self.bn3_3 = nn.BatchNorm1d(256)
        
        # Feature fusion layer
        self.fusion_conv = nn.Conv1d(768, 512, kernel_size=3, padding=1)  # 256*3=768
        self.fusion_bn = nn.BatchNorm1d(512)
        
        # Deep feature extraction
        self.deep_conv1 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.deep_conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.deep_bn1 = nn.BatchNorm1d(256)
        self.deep_bn2 = nn.BatchNorm1d(128)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        
        # Pooling layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Dropout layers (enhance regularization)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)
        
        # Classification layer (multi-scale feature classification)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # 128*2=256 (avg+max pooling)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, max_length, vocab_size)
        x = x.transpose(1, 2)  # (batch_size, vocab_size, max_length)
        
        # Branch 1: (4x21, 4x1, 16x1) - Small scale fine features
        branch1 = torch.relu(self.bn1_1(self.branch1_conv1(x)))
        branch1 = self.dropout1(branch1)
        branch1 = torch.relu(self.bn1_2(self.branch1_conv2(branch1)))
        branch1 = self.dropout2(branch1)
        branch1 = torch.relu(self.bn1_3(self.branch1_conv3(branch1)))
        branch1 = self.dropout1(branch1)
        
        # Branch 2: (12x21, 8x1, 4x1) - Medium scale features
        branch2 = torch.relu(self.bn2_1(self.branch2_conv1(x)))
        branch2 = self.dropout1(branch2)
        branch2 = torch.relu(self.bn2_2(self.branch2_conv2(branch2)))
        branch2 = self.dropout2(branch2)
        branch2 = torch.relu(self.bn2_3(self.branch2_conv3(branch2)))
        branch2 = self.dropout1(branch2)
        
        # Branch 3: (16x21, 4x1, 4x1) - Large scale global features
        branch3 = torch.relu(self.bn3_1(self.branch3_conv1(x)))
        branch3 = self.dropout1(branch3)
        branch3 = torch.relu(self.bn3_2(self.branch3_conv2(branch3)))
        branch3 = self.dropout2(branch3)
        branch3 = torch.relu(self.bn3_3(self.branch3_conv3(branch3)))
        branch3 = self.dropout1(branch3)
        
        # Multi-scale feature fusion
        x = torch.cat([branch1, branch2, branch3], dim=1)  # (batch_size, 768, max_length)
        x = torch.relu(self.fusion_bn(self.fusion_conv(x)))
        x = self.dropout3(x)
        
        # Deep feature extraction
        x = torch.relu(self.deep_bn1(self.deep_conv1(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.deep_bn2(self.deep_conv2(x)))
        x = self.dropout1(x)
        
        # Attention mechanism
        x = x.transpose(1, 2)  # (batch_size, max_length, 128)
        x_att, _ = self.attention(x, x, x)
        x = x + x_att  # Residual connection
        x = x.transpose(1, 2)  # (batch_size, 128, max_length)
        
        # Dual pooling
        x_avg = self.global_avg_pool(x)  # (batch_size, 128, 1)
        x_max = self.global_max_pool(x)  # (batch_size, 128, 1)
        x = torch.cat([x_avg, x_max], dim=1)  # (batch_size, 256, 1)
        x = x.squeeze(-1)  # (batch_size, 256)
        
        # Classification
        x = self.classifier(x)
        return x

def parse_fasta(fasta_file):
    """Parse FASTA file"""
    sequences = []
    current_header = None
    current_sequence = ""
    
    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append({
                        'header': current_header,
                        'sequence': current_sequence
                    })
                current_header = line[1:]
                current_sequence = ""
            else:
                current_sequence += line
        
        if current_header is not None:
            sequences.append({
                'header': current_header,
                'sequence': current_sequence
            })
    
    return sequences



def sequence_to_onehot(sequence, max_length=1000, vocab_size=21):
    """Convert amino acid sequence to one-hot encoding"""
    # Initialize one-hot matrix
    onehot = np.zeros((max_length, vocab_size), dtype=np.float32)
    
    # Encode sequence
    for i, aa in enumerate(sequence[:max_length]):
        aa_idx = aa_to_idx.get(aa, aa_to_idx['X'])
        onehot[i, aa_idx] = 1.0
    
    # For sequences shorter than max_length, remaining positions stay zero (padding)
    return onehot

def predict_sequences_batch(model, sequences, device, max_length=1000, batch_size=16):
    """Batch predict sequences"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Encode sequences as One-Hot
            encoded_batch = []
            for seq_data in batch_sequences:
                onehot_encoded = sequence_to_onehot(seq_data['sequence'], max_length)
                encoded_batch.append(onehot_encoded)
            
            # Convert to tensor
            batch_tensor = torch.tensor(np.stack(encoded_batch), dtype=torch.float).to(device)
            
            # Predict
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Extract probabilities
            for j, seq_data in enumerate(batch_sequences):
                non_tf_prob = probabilities[j][0].item()
                tf_prob = probabilities[j][1].item()
                
                all_predictions.append({
                    'header': seq_data['header'],
                    'tf_probability': tf_prob,
                    'non_tf_probability': non_tf_prob,
                    'sequence': seq_data['sequence'] # Add sequence for Grad-CAM
                })
    
    return all_predictions



def get_sequence_fragments(sequence, window_size=1000, step_size=200, mode='fast'):
    """Split sequence into fragments based on mode"""
    length = len(sequence)
    if length <= window_size:
        return [sequence]
    
    fragments = []
    
    if mode == 'full':
        # Sliding window with overlap (Step size 200)
        for start in range(0, length, step_size):
            end = start + window_size
            # If the window goes beyond the end, take from start to the end and stop
            if end >= length:
                fragments.append(sequence[start:])
                break
            fragments.append(sequence[start:end])
            
    elif mode == 'fast':
        # Fast mode: Contiguous 1000aa chunks + one final chunk from the end
        # Example: 1300 -> [0:1000], [300:1300]
        # Example: 3200 -> [0:1000], [1000:2000], [2000:3000], [2200:3200]
        
        # 1. Extract non-overlapping chunks
        num_chunks = length // window_size
        for i in range(num_chunks):
            start = i * window_size
            end = start + window_size
            fragments.append(sequence[start:end])
            
        # 2. Handle remainder by taking the last window_size characters
        if length % window_size != 0:
            fragments.append(sequence[-window_size:])
            
    return fragments

def predict_fasta(fasta_file=None, model=None, device=None, threshold=0.1, max_length=1000, batch_size=16, sequences=None, mode='fast', grad_cam_mode='none', grad_cam_output_dir=None, use_supplementary=False, supplementary_only=False, supp_model_paths=None):
    if sequences is None:
        print(f"\nProcessing file: {fasta_file}")
        sequences = parse_fasta(fasta_file)
        print(f"Number of sequences: {len(sequences)}")
    if not sequences:
        print("Warning: No valid sequences found")
        return [], []
    
    # Initialize Grad-CAM if needed
    grad_cam = None
    if (not supplementary_only) and model is not None and grad_cam_mode != 'none' and grad_cam_output_dir:
        os.makedirs(grad_cam_output_dir, exist_ok=True)
        try:
            grad_cam = GradCAM(model, target_layer_name='deep_conv2')
            print(f"Grad-CAM enabled. Output directory: {grad_cam_output_dir}")
        except Exception as e:
            print(f"Failed to initialize Grad-CAM: {e}")
    
    # Pre-process sequences: Split long sequences into fragments
    expanded_sequences = []
    fragment_counts = []  # Record number of fragments for each original sequence
    
    split_seq_count = 0 # Count of sequences that were split (length > window_size)
    total_fragments = 0 # Total number of fragments generated
    
    print(f"Sequence Splitting Mode: {mode}")
    
    for seq_data in sequences:
        fragments = get_sequence_fragments(seq_data['sequence'], window_size=max_length, step_size=200, mode=mode)
        num_frags = len(fragments)
        
        if num_frags > 1:
            split_seq_count += 1
        
        total_fragments += num_frags
        fragment_counts.append(num_frags)
        for frag in fragments:
            expanded_sequences.append({
                'header': seq_data['header'],
                'sequence': frag
            })
            
    print(f"Sequence Splitting Statistics:")
    print(f"  - Original Sequences: {len(sequences)}")
    print(f"  - Sequences Split (> {max_length}aa): {split_seq_count}")
    print(f"  - Total Fragments Generated: {total_fragments}")
    print(f"  - Average Fragments per Sequence: {total_fragments/len(sequences):.2f}")

    # Supplementary-only path
    if supplementary_only:
        supp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supplementary_model")
        paths = []
        if supp_model_paths:
            toks = []
            for p in supp_model_paths:
                toks.extend([x for x in p.split(',') if x.strip()])
            for p in toks:
                p2 = p
                if not os.path.isabs(p2):
                    p2 = os.path.join(supp_dir, p2)
                if os.path.exists(p2) and p2.endswith(".pth"):
                    paths.append(p2)
        else:
            if os.path.isdir(supp_dir):
                for name in os.listdir(supp_dir):
                    p = os.path.join(supp_dir, name)
                    if os.path.isfile(p) and p.endswith(".pth"):
                        paths.append(p)
        paths = sorted(list(set(paths)))
        if not paths:
            print(f"Error: No supplementary model files found in {supp_dir}")
            return [], []
        print("[Supplementary-Only] Models to use:")
        for p in paths:
            print(f"  - {p}")
        try:
            best_tf = [(-1.0, 0.0) for _ in range(len(expanded_sequences))]
            for model_path in paths:
                model_supp = OneHotLightweightCNN(vocab_size=21, num_classes=2, max_length=max_length)
                model_supp.load_state_dict(torch.load(model_path, map_location=device))
                model_supp.to(device)
                model_supp.eval()
                preds = predict_sequences_batch(model_supp, expanded_sequences, device, max_length, batch_size)
                for i, pred in enumerate(preds):
                    tfp = pred['tf_probability']
                    nfp = pred['non_tf_probability']
                    if tfp > best_tf[i][0]:
                        best_tf[i] = (tfp, nfp)
            results = []
            tf_headers = []
            idx = 0
            for count in fragment_counts:
                frag_scores = best_tf[idx:idx+count]
                idx += count
                best_idx = max(range(len(frag_scores)), key=lambda k: frag_scores[k][0])
                tf_prob = frag_scores[best_idx][0]
                non_prob = frag_scores[best_idx][1]
                seq_info = expanded_sequences[idx - count + best_idx]
                predicted_class = 'TF' if tf_prob >= threshold else 'Non-TF'
                confidence = max(tf_prob, non_prob)
                result = {
                    'header': seq_info['header'],
                    'predicted_class': predicted_class,
                    'tf_probability': tf_prob,
                    'non_tf_probability': non_prob,
                    'confidence': confidence,
                    'sequence': seq_info['sequence'],
                    'note': 'Supplementary'
                }
                results.append(result)
                if predicted_class == 'TF':
                    tf_headers.append(seq_info['header'])
            tf_count = sum(1 for p in results if p['predicted_class'] == 'TF')
            print(f"Prediction results: {tf_count} TF, {len(results)-tf_count} Non-TF")
            print(f"TF ratio: {tf_count/len(results)*100:.2f}%")
            return results, tf_headers
        except Exception as e:
            print(f"[Supplementary-Only] Error during prediction: {e}")
            return [], []
 
    # Predict on all fragments (main model)
    raw_predictions = predict_sequences_batch(model, expanded_sequences, device, max_length, batch_size)
    
    results = []
    tf_headers = []
    tf_count = 0
    
    # Aggregate results
    current_idx = 0
    
    # Initialize ProcessPoolExecutor for parallel plotting
    # Use a reasonable number of workers (e.g., 4) to avoid OOM or CPU contention
    # We only initialize it if Grad-CAM is actually enabled AND not in fast mode (which runs later)
    plot_executor = None
    if grad_cam and grad_cam_mode != 'fast':
        plot_executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
        
    # List to store TFs for fast mode (post-processing)
    fast_mode_tasks = []
        
    for count in fragment_counts:
        # Get predictions for all fragments of this protein
        frag_preds = raw_predictions[current_idx : current_idx + count]
        current_idx += count
        
        # Strategy: Max Pooling
        # If any fragment is predicted as TF (prob >= threshold), the protein is TF.
        # We take the fragment with the highest TF probability as the representative.
        best_pred = max(frag_preds, key=lambda x: x['tf_probability'])
        
        tf_prob = best_pred['tf_probability']
        predicted_class = 'TF' if tf_prob >= threshold else 'Non-TF'
        confidence = max(tf_prob, best_pred['non_tf_probability'])
        
        result = {
            'header': best_pred['header'],
            'predicted_class': predicted_class,
            'tf_probability': tf_prob,
            'non_tf_probability': best_pred['non_tf_probability'],
            'confidence': confidence,
            'sequence': best_pred['sequence'] # Keep sequence for later use
        }
        results.append(result)

    # --- Supplementary Model Logic ---
    supp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supplementary_model")
    supplementary_tfs = []
    
    if use_supplementary:
        paths = []
        if supp_model_paths:
            toks = []
            for p in supp_model_paths:
                toks.extend([x for x in p.split(',') if x.strip()])
            for p in toks:
                p2 = p
                if not os.path.isabs(p2):
                    p2 = os.path.join(supp_dir, p2)
                if os.path.exists(p2) and p2.endswith(".pth"):
                    paths.append(p2)
        else:
            if os.path.isdir(supp_dir):
                for name in os.listdir(supp_dir):
                    p = os.path.join(supp_dir, name)
                    if os.path.isfile(p) and p.endswith(".pth"):
                        paths.append(p)
        paths = sorted(list(set(paths)))
        if not paths:
            print("[Supplementary] No supplementary model files found; skipping")
        else:
            print(f"\n[Supplementary] Using models:")
            for p in paths:
                print(f"  - {p}")
            print("[Supplementary] Re-evaluating negative predictions...")
        
        try:
            best_tf = None
            
            # Identify negatives
            negatives_indices = [i for i, res in enumerate(results) if res['predicted_class'] == 'Non-TF']
            
            if negatives_indices:
                # Reconstruct fragments for negative proteins
                supp_fragments = []
                supp_mapping = [] # (result_index, start_idx, count)
                
                # We need to traverse fragment_counts again to get correct slices from expanded_sequences
                curr_frag_idx = 0
                for i, count in enumerate(fragment_counts):
                    if i in negatives_indices:
                        # This protein was negative. Add its fragments to supp batch.
                        frags = expanded_sequences[curr_frag_idx : curr_frag_idx + count]
                        supp_start = len(supp_fragments)
                        supp_fragments.extend(frags)
                        supp_mapping.append((i, supp_start, count))
                    curr_frag_idx += count
                
                if supp_fragments:
                    print(f"[Supplementary] Predicting on {len(supp_fragments)} fragments from {len(negatives_indices)} sequences...")
                    best_tf = [(-1.0, 0.0) for _ in range(len(supp_fragments))]
                    for model_path in paths:
                        model_supp = OneHotLightweightCNN(vocab_size=21, num_classes=2, max_length=max_length)
                        model_supp.load_state_dict(torch.load(model_path, map_location=device))
                        model_supp.to(device)
                        model_supp.eval()
                        supp_preds = predict_sequences_batch(model_supp, supp_fragments, device, max_length, batch_size)
                        for i_pred, pred in enumerate(supp_preds):
                            tfp = pred['tf_probability']
                            nfp = pred['non_tf_probability']
                            if tfp > best_tf[i_pred][0]:
                                best_tf[i_pred] = (tfp, nfp)
                    
                    rescued_count = 0
                    for res_idx, supp_start, count in supp_mapping:
                        frag_scores = best_tf[supp_start : supp_start + count]
                        best_idx = max(range(len(frag_scores)), key=lambda k: frag_scores[k][0])
                        tf_prob = frag_scores[best_idx][0]
                        non_prob = frag_scores[best_idx][1]
                        
                        if tf_prob >= threshold:
                            results[res_idx]['predicted_class'] = 'TF'
                            results[res_idx]['tf_probability'] = tf_prob
                            results[res_idx]['non_tf_probability'] = non_prob
                            results[res_idx]['confidence'] = max(tf_prob, non_prob)
                            results[res_idx]['note'] = 'Supplementary'
                            
                            supplementary_tfs.append(results[res_idx])
                            rescued_count += 1
                            
                    print(f"[Supplementary] Rescued {rescued_count} sequences.")
            else:
                print("[Supplementary] No negative predictions to re-evaluate.")
                
        except Exception as e:
            print(f"[Supplementary] Error during supplementary prediction: {e}")
            import traceback
            traceback.print_exc()

    # --- End Supplementary Logic ---

    # Final Loop to collect TFs and generate Grad-CAM
    for i, result in enumerate(results):
        predicted_class = result['predicted_class']
        
        if predicted_class == 'TF':
            tf_count += 1
            tf_headers.append(result['header'])
            
            # If fast mode, save for later
            if grad_cam and grad_cam_mode == 'fast':
                # We need to reconstruct the best_pred object for fast_mode_tasks
                # The result has most info, but fast_mode_tasks expects a dict with 'sequence'
                # which we added to result above.
                fast_mode_tasks.append(result)
            
        # Generate Grad-CAM based on mode (Real-time modes)
        should_generate_gradcam = False
        if grad_cam and grad_cam_mode != 'fast':
            if grad_cam_mode == 'all':
                should_generate_gradcam = True
            elif grad_cam_mode == 'positive' and predicted_class == 'TF':
                should_generate_gradcam = True
                
        if should_generate_gradcam:
            try:
                # Prepare input tensor for the best fragment
                seq = result['sequence']
                onehot = sequence_to_onehot(seq, max_length=max_length)
                input_tensor = torch.tensor(onehot, dtype=torch.float).unsqueeze(0).to(device)
                input_tensor.requires_grad = True
                
                # If predicted as Non-TF, we might still want to see TF activation (target_class=1)
                # or the predicted class. For consistency with previous 'positive' only behavior (target_class=1),
                # we will stick to target_class=1 (TF) to see TF-relevant features, 
                # but if you prefer seeing what the model predicted, use target_class=None.
                # Here we use target_class=1 to consistently visualize "TF-ness".
                cam, output, _ = grad_cam.generate(input_tensor, target_class=1) 
                
                if cam is not None:
                    header = result['header']
                    safe_prefix = "".join([c if c.isalnum() else "_" for c in header])[:60].strip("_")
                    header_hash = hashlib.md5(header.encode('utf-8')).hexdigest()[:8]
                    safe_header = f"{safe_prefix}_{header_hash}_{i}"
                    save_path = os.path.join(grad_cam_output_dir, f"{safe_header}.png")
                    
                    # Submit plotting task to executor
                    if plot_executor:
                        plot_executor.submit(plot_gradcam, seq, cam, f"{header}", save_path)
                    else:
                        # Fallback to synchronous execution if executor failed (shouldn't happen)
                        plot_gradcam(seq, cam, f"{header}", save_path)
                        
            except Exception as e:
                print(f"Error generating Grad-CAM for {result.get('header', 'unknown')}: {e}")
            
    # Clean up executor (Real-time modes)
    if plot_executor:
        # We print a message so user knows what's happening if it hangs a bit
        # print("Waiting for background Grad-CAM plots to finish...") 
        plot_executor.shutdown(wait=True)
        
    # Handle Fast Mode (Post-processing)
    if grad_cam and grad_cam_mode == 'fast' and fast_mode_tasks:
        print(f"Starting Fast Mode Grad-CAM generation for {len(fast_mode_tasks)} TF sequences...")
        
        # We can use ProcessPoolExecutor for plotting here as well
        plot_executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
        
        try:
            for i, best_pred in enumerate(fast_mode_tasks):
                try:
                    # Prepare input tensor
                    seq = best_pred['sequence']
                    onehot = sequence_to_onehot(seq, max_length=max_length)
                    input_tensor = torch.tensor(onehot, dtype=torch.float).unsqueeze(0).to(device)
                    input_tensor.requires_grad = True
                    
                    # Generate Grad-CAM (Forward + Backward)
                    # Note: This runs on the main thread/GPU, which is fine as it's sequential
                    # but plotting is offloaded
                    cam, output, _ = grad_cam.generate(input_tensor, target_class=1) 
                    
                    if cam is not None:
                        header = best_pred['header']
                        safe_prefix = "".join([c if c.isalnum() else "_" for c in header])[:60].strip("_")
                        header_hash = hashlib.md5(header.encode('utf-8')).hexdigest()[:8]
                        safe_header = f"{safe_prefix}_{header_hash}_{i}"
                        save_path = os.path.join(grad_cam_output_dir, f"{safe_header}.png")
                        
                        # Submit plotting task
                        plot_executor.submit(plot_gradcam, seq, cam, f"{header}", save_path)
                        
                except Exception as e:
                    print(f"Error generating Grad-CAM for {best_pred['header']}: {e}")
                    
                # Optional: Print progress
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(fast_mode_tasks)} sequences")
                    
        finally:
             plot_executor.shutdown(wait=True)
        print("Fast Mode Grad-CAM generation completed.")

    print(f"Prediction results: {tf_count} TF, {len(results) - tf_count} Non-TF")
    print(f"TF ratio: {tf_count/len(results)*100:.2f}%")
    return results, tf_headers

def save_predictions(predictions, output_file):
    """Save prediction results to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Header', 'Predicted_Class', 'TF_Probability', 'Non_TF_Probability', 'Confidence'])
        for pred in predictions:
            writer.writerow([
                pred['header'],
                pred['predicted_class'],
                f"{pred['tf_probability']:.4f}",
                f"{pred['non_tf_probability']:.4f}",
                f"{pred['confidence']:.4f}"
            ])
    print(f"Prediction results saved to: {output_file}")

def save_tf_predictions(predictions, output_file):
    """Save TF prediction results to CSV file"""
    tf_predictions = [p for p in predictions if p['predicted_class'] == 'TF']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Header', 'TF_Probability', 'Confidence'])
        for pred in tf_predictions:
            writer.writerow([
                pred['header'],
                f"{pred['tf_probability']:.4f}",
                f"{pred['confidence']:.4f}"
            ])
    print(f"TF prediction results saved to: {output_file}")

def load_model(model_path, device, max_length=1000):
    """Load trained multi-scale CNN model"""
    print(f"Loading multi-scale CNN model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Accept both the full checkpoint dict (with 'model_config'/'model_state_dict')
    # and a bare state_dict (as written by supplementary_model/train.py, which
    # saves model.state_dict() directly). Fall back to the checkpoint itself.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Get model configuration
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        config = checkpoint['model_config']
        vocab_size = config.get('vocab_size', 21)
        max_length = config.get('max_length', 1000)
        num_classes = config.get('num_classes', 2)
    else:
        # Default configuration
        vocab_size = 21
        num_classes = 2

    # Create multi-scale CNN model
    model = OneHotLightweightCNN(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_length=max_length
    ).to(device)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Multi-scale CNN model loaded successfully:")
    print(f"  - Model Architecture: Three-branch parallel multi-scale CNN")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Vocabulary Size: {vocab_size}")
    print(f"  - Max Sequence Length: {max_length}")
    print(f"  - Number of Classes: {num_classes}")
    if 'val_acc' in checkpoint:
        print(f"  - Validation Accuracy: {checkpoint['val_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  - Training Epochs: {checkpoint['epoch'] + 1}")
    
    return model, max_length

def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_project_output_dir(fasta_file):
    root = _repo_root()
    output_base = os.path.join(root, "output")
    base = os.path.splitext(os.path.basename(fasta_file))[0]
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    candidates = []
    for name in os.listdir(output_base):
        if name == base or name.startswith(base + "_"):
            full = os.path.join(output_base, name)
            if os.path.isdir(full):
                candidates.append(full)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    target = os.path.join(output_base, base)
    os.makedirs(target, exist_ok=True)
    return target

def main():
    # Get absolute path of script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, 'model.pth')
    
    parser = argparse.ArgumentParser(description='Multi-scale CNN Model Protein Prediction')
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file path')
    parser.add_argument('--model', type=str, 
                        default=default_model_path,
                        help=f'Model file path (default: {default_model_path})')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='TF prediction threshold (default: 0.1, i.e., 10%%)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated based on input filename)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode, generate CSV file (default: False)')
    parser.add_argument('--output-tf-list', action='store_true',
                       help='Output TF list to stdout for memory passing (default: False)')
    
    parser.add_argument('--project-output', type=str, default=None,
                       help='Project output directory path')
    parser.add_argument('--mode', type=str, choices=['fast', 'full'], default='fast',
                       help='Sequence splitting mode: fast (contiguous chunks + tail) or full (sliding window). Default: fast')
    
    # Grad-CAM arguments
    parser.add_argument('--grad-cam-mode', type=str, choices=['none', 'fast', 'all', 'positive'], default='none',
                       help='Grad-CAM visualization mode: none (default), fast (post-processing TFs only), all (all sequences), positive (real-time TFs)')
    parser.add_argument('--grad-cam-output', type=str, default=None,
                       help='Output directory for Grad-CAM images. Defaults to output_dir/gradcam')
    
    parser.add_argument('--use-supplementary', action='store_true', default=False,
                       help='Enable supplementary model for cascaded prediction (default: False)')
     
    parser.add_argument('--supplementary-only', action='store_true', default=False,
                        help='Use only the supplementary model for prediction and skip main model')
     
    parser.add_argument('--supp-models', type=str, nargs='*', default=None,
                        help='Specify supplementary model filenames or paths; if omitted, use all in supplementary_model')
    parser.add_argument('--num-threads', type=int, default=0, dest='num_threads',
                        help='Number of CPU threads for PyTorch (torch.set_num_threads). 0 = use default (default: 0)')

    args = parser.parse_args()

    if args.num_threads and args.num_threads > 0:
        try:
            torch.set_num_threads(args.num_threads)
        except Exception as e:
            print(f"[WARN] Failed to set torch num_threads={args.num_threads}: {e}")
    
    # Check input file
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA file not found: {args.fasta}")
        return
    
    # Check model file (skip if supplementary-only)
    if not args.supplementary_only:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            return
    
    # Generate output filename
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.fasta))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{base_name}_predictions_{timestamp}.csv"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if args.supplementary_only:
        model, max_length = None, 1000
    else:
        model, max_length = load_model(args.model, device)
    
    input_name = os.path.splitext(os.path.basename(args.fasta))[0]

    # Determine Grad-CAM output directory
    grad_cam_dir = None
    if args.grad_cam_mode != 'none':
        if args.grad_cam_output:
            grad_cam_dir = args.grad_cam_output
        elif args.project_output:
            grad_cam_dir = os.path.join(args.project_output, f"{input_name}_gradcam")
        elif args.output:
            # If output is a file path, put gradcam in a folder next to it
            output_dir = os.path.dirname(os.path.abspath(args.output))
            grad_cam_dir = os.path.join(output_dir, f"{input_name}_gradcam")
        else:
            grad_cam_dir = f"{input_name}_gradcam"
            
        # Ensure directory exists
        if grad_cam_dir:
            os.makedirs(grad_cam_dir, exist_ok=True)
    
    # Predict
    repo_root = os.path.dirname(script_dir)
    sys.path.append(repo_root)
    try:
        from module.get_fasta import generate_protein_sequences_in_memory
        sequences = generate_protein_sequences_in_memory(args.fasta)
        predictions, tf_headers = predict_fasta(
            model=model,
            device=device,
            threshold=args.threshold,
            max_length=max_length,
            batch_size=args.batch_size,
            sequences=sequences,
            mode=args.mode,
            grad_cam_mode=args.grad_cam_mode,
            grad_cam_output_dir=grad_cam_dir,
            use_supplementary=args.use_supplementary,
            supplementary_only=args.supplementary_only,
            supp_model_paths=args.supp_models
        )
    except Exception:
        predictions, tf_headers = predict_fasta(
            fasta_file=args.fasta,
            model=model,
            device=device,
            threshold=args.threshold,
            max_length=max_length,
            batch_size=args.batch_size,
            mode=args.mode,
            grad_cam_mode=args.grad_cam_mode,
            grad_cam_output_dir=grad_cam_dir,
            use_supplementary=args.use_supplementary,
            supplementary_only=args.supplementary_only,
            supp_model_paths=args.supp_models
        )
    
    if predictions:
        # Save prediction results (Always save in normal mode too)
        # Save all prediction results
        save_predictions(predictions, args.output)
        
        # Save TF prediction results
        tf_output = args.output.replace('.csv', '_tf_only.csv')
        save_tf_predictions(predictions, tf_output)

        # Output TF list to stdout (for memory passing)
        if args.output_tf_list:
            tf_headers = [p['header'] for p in predictions if p['predicted_class'] == 'TF']
            print("TF_LIST_START")
            for header in tf_headers:
                print(f"TF_HEADER:{header}")
            print("TF_LIST_END")

        # Collect supplementary TFs if any
        supplementary_tfs = [p for p in predictions if p.get('note') == 'Supplementary']

        if supplementary_tfs:
            pass

        print(f"\n=== Prediction Completed ===")
        print(f"Total Sequences: {len(predictions)}")
        tf_count = sum(1 for p in predictions if p['predicted_class'] == 'TF')
        print(f"Predicted TF Count: {tf_count}")
        print(f"TF Ratio: {tf_count/len(predictions)*100:.2f}%")
        print(f"Threshold Used: {args.threshold} ({args.threshold*100:.1f}%)")
    else:
        print("Error: Prediction failed or no valid sequences found")

if __name__ == '__main__':
    main()
