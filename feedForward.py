import torch, torch.nn as nn

batch_size = 64
max_len = 256
d_model = 384
n_head = 6
d_q = int(d_model / n_head) 
dropout = 0.2

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, 4*d_model), # Expand dimension, so when applying ReLU, you mitigate loss of information issue
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.seq(x)

if __name__ == '__main__':
    x = torch.randn(batch_size, max_len, d_model)
    ffwd = FeedForward(d_model)
    output = ffwd(x)

    print("Input shape:", x.shape)
    print("Output shape from FeedForward network:", output.shape)