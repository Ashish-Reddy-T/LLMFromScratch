import torch, torch.nn as nn, torch.nn.functional as F

batch_size = 64
max_len = 256
d_model = 384
n_layer = 6 # 6 blocks in the decoder
n_head = 6
d_q = int(d_model / n_head) 
dropout = 0.2
vocab_size = 65

from block import Block

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model) # Embedding matrix size: (65, 384)
        self.positional_embedding_table = nn.Embedding(max_len, d_model) # Position matrix size: (256, 384)
        self.blocks = nn.Sequential(*[Block(d_model, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.unembedding_matrix_calc = nn.Linear(d_model, vocab_size)
    
    def forward(self, idx, targets=None):
        B, S = idx.shape

        tok_emb = self.token_embedding_table(idx) # Size of embedding: (B, S, 384)
        pos_emb = self.positional_embedding_table(torch.arange(S, device=idx.device)) # Shape: (S, 384)
        x = tok_emb + pos_emb

        x = self.blocks(x) # Pass through all 6 blocks each of all 6 heads
        x = self.ln(x)

        logits = self.unembedding_matrix_calc(x) # --> (B, S, 384) * (384, 65) --> (B, S, 65)

        if targets is None:
            loss = None
        else:
            B, S, V = logits.shape
            logits = logits.view(-1, V) # (B, S, V) --> (B*S, V)
            targets = targets.view(-1) # --> (B, S) --> (B*S)
            loss = F.cross_entropy(logits, targets) # Handles softmax interally as well (better because it does log addition which reduces errors instead of log multi)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len:]

if __name__ == "__main__":
    model = Model()
    idx = torch.zeros((batch_size, max_len), dtype=torch.long)
    logits, loss = model(idx, idx)

    print("Input shape:", idx.shape)
    print("Output logits shape:", logits.shape)
    print("Calculated loss:", loss.item())