import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=20):

        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=64, num_heads=4):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) 
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention_weights, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(out)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=64, d_ff=128, dropout_rate=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=4, d_ff=128, dropout_rate=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_out = self.attention(x, mask)
        x = self.layer_norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_out))
        return x
    
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128, num_layers=1, max_len=20, dropout_rate=0.1, use_pe=True):
        super().__init__()
        self.d_model = d_model
        self.use_pe = use_pe
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, tokens, mask):
        x = self.token_emb(tokens) * math.sqrt(self.d_model)
        if self.use_pe:
            x = self.pos_emb(x)
    
        for block in self.blocks:
            x = block(x, mask)
            
        mask_expanded = mask.unsqueeze(-1) 
        
        x_masked = x * mask_expanded

        summed_x = torch.sum(x_masked, dim=1)
        
        valid_counts = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)

        pooled_x = summed_x / valid_counts
        
        logits = self.classifier(pooled_x).squeeze(-1) 
        
        return logits
