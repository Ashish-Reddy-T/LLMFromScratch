import torch, torch.nn as nn

batch_size = 64
max_len = 256
d_model = 384
n_head = 6
d_q = int(d_model / n_head) 
dropout = 0.2

from multiHead import MultiHead
from feedForward import FeedForward

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.multiHead = MultiHead(n_head, d_q)
        self.ffwd = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        normalized1 = self.ln1(x) # It is seen that pre LN works better as it stabilizes better
        multiHead = self.multiHead(normalized1)
        x = x + multiHead # Residual Addition

        normalized2 = self.ln2(x)
        ffwd = self.ffwd(normalized2)
        x = x + ffwd

        return x

if __name__ == "__main__":
    x = torch.randn(batch_size, max_len, d_model)
    block = Block(d_model, n_head)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape from one Transformer Block:", output.shape)