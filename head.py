import torch, torch.nn as nn, torch.nn.functional as F

batch_size = 64
max_len = 256
d_model = 384
n_head = 6
d_q = int (d_model / n_head) # 384/6 = 64
dropout = 0.2

class Head(nn.Module):
    def __init__(self, d_q):
        super().__init__()
        self.query = nn.Linear(d_model, d_q, bias=False) # Query weight matrix (Wq) = Linear, pass in x with shape (seq, 384) * (384, 64) to get q = (seq, 64) size
        self.key = nn.Linear(d_model, d_q, bias=False)   # k = x * Wk
        self.value = nn.Linear(d_model, d_q, bias=False) # v = x * Wv

        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len))) # Save it to register_buffer, as a non-trainable parameter / buffer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, S, D = x.shape # B --> Batch; S --> Seq_length; D --> Dimension

        q = self.query(x) # Shape of q: (Batch, Seq_Len, d_q) = (B, S, 64)
        k = self.key(x) 
        v = self.value(x)

        attention_matrix = torch.matmul(q, k.transpose(-2, -1)) # --> (B, S, 64) * (B, 64, S) --> (B, S, S) shape
        attention_matrix = attention_matrix / (k.size(-1) ** 0.5)

        attention_matrix = attention_matrix.masked_fill(self.tril[:S, :S] == 0, float('-inf')) # Makes upper right triangle True because they are all 0s and all 1s (lower half of triangle) false and wherever it is True, fill it in with -inf or in other words fill the spots with 0s as -inf so as we are creating a causal decoder that isn't bidirectional

        attention_matrix = F.softmax(attention_matrix, dim=-1) # dim = -1, to apply softmax row-wise
        attention_matrix = self.dropout(attention_matrix) # Apply 20% dropout to prevent overfitting
        output = torch.matmul(attention_matrix, v) # --> (B, S, S) * (B, S, 64) --> (B, S, 64) (Original x dimension after concat, so you can now simply add)

        return output

if __name__ == "__main__":
    x = torch.randn(batch_size, max_len, d_model)
    single_head = Head(d_q)
    output = single_head(x)

    print("Input shape:", x.shape)
    print("Output shape from a single head:", output.shape)