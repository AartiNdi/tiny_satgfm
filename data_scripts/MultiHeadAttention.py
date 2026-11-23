import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim needs to be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        # store the arguments
        # check that embed_dim is divisible by num_heads
        # compute head_dim = embed_dim // num_heads

        # create the necessary layers:
        # - one linear layer that outputs 3 * embed_dim (for Q, K, V together)
        self.kqv = nn.Linear(embed_dim, 3*embed_dim)
        # - one linear layer that projects the result back to embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
        # - a dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        B, N, C = x.shape

        # 1. Project x to Q, K, V
        qkv = self.kqv(x)  # shape: [B, N, 3*embed_dim]
        # 2. Reshape so we have separate heads
        #    target shape after reshape + permute: [3, B, num_heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 3. Unpack into q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]  # each shape: [B, num_heads, N, head_dim]

        # scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) 
        #transpose makes [Batch, num_heads, head_dim, N]. Shape of q is [B, num_heads, N, head_dim]
        # shape of product is [B, num_heads, N, N]
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v          # [B, num_heads, N, head_dim]
        # merge heads back: [B, N, embed_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        # final linear projection
        out = self.proj(out)
        return out
    