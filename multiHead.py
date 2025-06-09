import torch, torch.nn as nn

batch_size = 64
max_len = 256
d_model = 384
n_head = 6
d_q = int(d_model / n_head) 
dropout = 0.2

from head import Head

class MultiHead(nn.Module):
    def __init__(self, n_head, d_q):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_q) for _ in range(n_head)]) # Create a list of 6 heads with different randomized weights each
        self.proj = nn.Linear(d_model, d_model) # You concat your 6 heads to shape (B, S, 384) * (384, 384) --> (B, S, 384) (Ready to be added! Residual connection)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        concatenated_outputs = torch.cat([_(x) for _ in self.heads], dim=-1) # Concat each output (B, S, 64) horizontally to get (B, S, 384) as there are 6 heads
        output = self.proj(concatenated_outputs)
        output = self.dropout(output)
        return output

if __name__ == "__main__":
    x = torch.randn(batch_size, max_len, d_model)
    multi_head = MultiHead(n_head, d_q)
    output = multi_head(x)

    print("Input shape:", x.shape)
    print("Output shape from multi-head:", output.shape)