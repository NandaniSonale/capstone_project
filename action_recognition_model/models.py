import torch
import torch.nn as nn

class TemporalCNNActionClassifier(nn.Module):
    """
    Temporal Convolutional Network (Temporal 1D-CNN) for Compressed-Domain Action Recognition.
    Processes (B, 60, 8) motion sequences using dilated 1D temporal convolutions,
    multi-scale receptive fields, residual skip connections, and global pooling.
    """
    def __init__(self, input_dim=8, num_classes=7, dropout=0.3):
        super(TemporalCNNActionClassifier, self).__init__()
        
        # Block 1: Initial temporal feature expansion (kernel size 5)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        # Block 2: Multi-scale residual temporal block (dilation=1 and dilation=2)
        self.conv2a = nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1)
        self.bn2a = nn.BatchNorm1d(128)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.bn2b = nn.BatchNorm1d(128)
        self.res_proj2 = nn.Conv1d(64, 128, kernel_size=1)
        self.drop2 = nn.Dropout(dropout)
        
        # Block 3: Deep dilated temporal block (dilation=4 for broad temporal receptive field)
        self.conv3a = nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4)
        self.bn3a = nn.BatchNorm1d(256)
        self.conv3b = nn.Conv1d(256, 256, kernel_size=3, padding=1, dilation=1)
        self.bn3b = nn.BatchNorm1d(256)
        self.res_proj3 = nn.Conv1d(128, 256, kernel_size=1)
        self.drop3 = nn.Dropout(dropout)
        
        # Dual Global Temporal Pooling (Average + Max for sustained and peak motions)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification MLP head
        self.fc1 = nn.Linear(256 * 2, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, seq_len=60, input_dim=8)
        # Transpose for Conv1D: (batch_size, channels=8, seq_len=60)
        x = x.transpose(1, 2)
        
        # Block 1
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        
        # Block 2 with residual connection
        res2 = self.res_proj2(out)
        out = self.relu(self.bn2a(self.conv2a(out)))
        out = self.relu(self.bn2b(self.conv2b(out)))
        out = self.relu(out + res2)
        out = self.drop2(out)
        
        # Block 3 with residual connection
        res3 = self.res_proj3(out)
        out = self.relu(self.bn3a(self.conv3a(out)))
        out = self.relu(self.bn3b(self.conv3b(out)))
        out = self.relu(out + res3)
        out = self.drop3(out)
        
        # Dual Global Temporal Pooling
        avg_p = self.avg_pool(out).squeeze(-1)
        max_p = self.max_pool(out).squeeze(-1)
        pooled = torch.cat([avg_p, max_p], dim=1) # shape: (B, 512)
        
        # Classification Head
        feat = self.relu(self.bn_fc(self.fc1(pooled)))
        feat = self.drop_fc(feat)
        logits = self.fc_out(feat)
        return logits


class BiLSTMActionClassifier(nn.Module):
    """
    Bidirectional LSTM Deep Learning Architecture for Compressed-Domain Action Recognition.
    Processes (B, 60, 8) motion sequences across time.
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, num_classes=7, dropout=0.3):
        super(BiLSTMActionClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, seq_len=60, input_dim=8)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Global temporal average + max pooling
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled = (avg_pool + max_pool) / 2.0
        
        out = self.bn1(pooled)
        out = self.dropout1(out)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        logits = self.fc_out(out)
        return logits
