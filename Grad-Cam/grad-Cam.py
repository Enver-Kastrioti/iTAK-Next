
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Configure fonts for CJK glyph rendering when available
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# Model definition (copied from train_onehot_cnn.py)
# ---------------------------------------------------------

class OneHotLightweightCNN(nn.Module):
    """Multi-scale CNN model using One-Hot encoding.
    
    Uses three parallel convolutional branches with different kernel configurations for
    multi-scale feature extraction:
    Config 1: (4×21, 4×1, 16×1)
    Config 2: (12×21, 8×1, 4×1)
    Config 3: (16×21, 4×1, 4×1)
    """
    def __init__(self, vocab_size=21, num_classes=2, max_length=1000):
        super(OneHotLightweightCNN, self).__init__()
        
        # Input: (batch_size, max_length, vocab_size)
        # Transpose to: (batch_size, vocab_size, max_length) for Conv1d
        
        # Config 1: (4×21, 4×1, 16×1) - small-scale fine-grained features
        self.branch1_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=4, padding=2)  # equivalent to 4×21
        self.branch1_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)        # 4×1
        self.branch1_conv3 = nn.Conv1d(128, 256, kernel_size=16, padding=8)      # 16×1
        
        # Config 2: (12×21, 8×1, 4×1) - medium-scale features
        self.branch2_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=12, padding=6) # equivalent to 12×21
        self.branch2_conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)        # 8×1
        self.branch2_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)       # 4×1
        
        # Config 3: (16×21, 4×1, 4×1) - large-scale global features
        self.branch3_conv1 = nn.Conv1d(vocab_size, 64, kernel_size=16, padding=8) # equivalent to 16×21
        self.branch3_conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2)        # 4×1
        self.branch3_conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2)       # 4×1
        
        # Batch normalization layers - branch 1
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn1_3 = nn.BatchNorm1d(256)
        
        # Batch normalization layers - branch 2
        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(256)
        
        # Batch normalization layers - branch 3
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
        
        # Dropout layers (regularization)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)
        
        # Classification head
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
        
        # Branch 1: (4×21, 4×1, 16×1) - small-scale fine-grained features
        branch1 = torch.relu(self.bn1_1(self.branch1_conv1(x)))
        branch1 = self.dropout1(branch1)
        branch1 = torch.relu(self.bn1_2(self.branch1_conv2(branch1)))
        branch1 = self.dropout2(branch1)
        branch1 = torch.relu(self.bn1_3(self.branch1_conv3(branch1)))
        branch1 = self.dropout1(branch1)
        
        # Branch 2: (12×21, 8×1, 4×1) - medium-scale features
        branch2 = torch.relu(self.bn2_1(self.branch2_conv1(x)))
        branch2 = self.dropout1(branch2)
        branch2 = torch.relu(self.bn2_2(self.branch2_conv2(branch2)))
        branch2 = self.dropout2(branch2)
        branch2 = torch.relu(self.bn2_3(self.branch2_conv3(branch2)))
        branch2 = self.dropout1(branch2)
        
        # Branch 3: (16×21, 4×1, 4×1) - large-scale global features
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
            raise ValueError(f"Layer {self.target_layer_name} not found in model.")
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        
    def generate(self, input_tensor, target_class=None):
        # input_tensor: (1, max_len, vocab_size)
        self.model.eval()
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
        # Gradients: (batch, channels, length) -> (1, 128, 1000)
        # Activations: (batch, channels, length) -> (1, 128, 1000)
        
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

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
aa_to_idx['X'] = len(AMINO_ACIDS)

def sequence_to_onehot(sequence, max_length=1000):
    vocab_size = len(aa_to_idx)
    onehot = np.zeros((max_length, vocab_size), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_length]):
        aa_idx = aa_to_idx.get(aa, aa_to_idx['X'])
        onehot[i, aa_idx] = 1.0
    return onehot

def read_fasta(file_path):
    sequences = []
    current_seq = ""
    current_header = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences.append({'header': current_header, 'sequence': current_seq})
                current_header = line[1:]
                current_seq = ""
            else:
                current_seq += line
        if current_header:
             sequences.append({'header': current_header, 'sequence': current_seq})
    return sequences

def plot_gradcam(sequence, cam, title, save_path):
    # cam is 1D array of length 1000 (or model max_length)
    # sequence is string
    
    # Trim CAM to sequence length (if seq < max_length)
    seq_len = len(sequence)
    if seq_len > 1000:
        seq_len = 1000
        sequence = sequence[:1000]
        
    cam_trimmed = cam[:seq_len]
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
    # If sequence is too long, we might not want to print every char, but let's try
    # For very long sequences, maybe just plot the curve
    
    plt.plot(cam_trimmed, color='black', linewidth=1, alpha=0.5)
    plt.fill_between(range(seq_len), cam_trimmed, alpha=0.2, color='blue')
    
    plt.colorbar(label='Importance')
    plt.title(title)
    plt.xlabel('Sequence Position')
    plt.ylabel('Activation')
    
    # Add high activation residues text
    threshold = 0.6
    high_act_indices = np.where(cam_trimmed > threshold)[0]
    
    # Annotate top 5 peaks
    if len(high_act_indices) > 0:
        # Simple peak finding or just listing
        pass

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Prediction for Protein Sequences")
    parser.add_argument('--model', type=str, default=r'd:\newiTak\tool\domain_based_TF_binary\Grad-CAM\8.28model.pth', help='Path to model file')
    parser.add_argument('--fasta', type=str, default=r'd:\newiTak\tool\domain_based_TF_binary\Grad-CAM\Apr_pep.fasta', help='Path to FASTA file')
    parser.add_argument('--output_dir', type=str, default=r'd:\newiTak\tool\domain_based_TF_binary\Grad-CAM\gradcam_results', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Initialize model
    model_config = checkpoint.get('model_config', {})
    vocab_size = model_config.get('vocab_size', 21)
    max_length = model_config.get('max_length', 1000)
    num_classes = model_config.get('num_classes', 2)
    
    model = OneHotLightweightCNN(vocab_size=vocab_size, num_classes=num_classes, max_length=max_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer_name='deep_conv2')
    
    # Read Sequences
    print(f"Reading sequences from {args.fasta}")
    sequences = read_fasta(args.fasta)
    print(f"Found {len(sequences)} sequences.")
    
    results = []
    
    print("Processing sequences...")
    for idx, item in enumerate(sequences):
        header = item['header']
        seq = item['sequence']
        
        # Preprocess
        onehot = sequence_to_onehot(seq, max_length=max_length)
        input_tensor = torch.tensor(onehot, dtype=torch.float).unsqueeze(0).to(device) # (1, max_len, vocab_size)
        input_tensor.requires_grad = True
        
        # Predict & Grad-CAM
        cam, output, pred_class = grad_cam.generate(input_tensor)
        
        probs = F.softmax(output, dim=1)
        prob_pos = probs[0, 1].item()
        
        prediction = "YES" if prob_pos > 0.5 else "NO"
        
        print(f"[{idx+1}/{len(sequences)}] {header[:20]}... -> Class: {pred_class} ({prediction}), Prob: {prob_pos:.4f}")
        
        results.append({
            'header': header,
            'prediction': prediction,
            'probability': prob_pos
        })
        
        # If positive prediction (or user wants all), save heatmap
        # For demonstration, let's save if prob > 0.5
        if prob_pos > 0.5:
            safe_header = "".join([c if c.isalnum() else "_" for c in header[:30]])
            save_path = os.path.join(args.output_dir, f"gradcam_{idx}_{safe_header}.png")
            plot_gradcam(seq, cam, f"Grad-CAM: {header[:50]}...", save_path)
            
            # Also save raw cam data
            np.save(os.path.join(args.output_dir, f"gradcam_{idx}_{safe_header}.npy"), cam)
            
    # Save summary results
    import csv
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Header', 'Prediction', 'Probability'])
        for res in results:
            writer.writerow([res['header'], res['prediction'], res['probability']])
            
    print(f"\nDone! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
