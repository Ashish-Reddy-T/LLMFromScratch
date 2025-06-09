import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, wandb, time

batch_size = 64
max_len = 256
d_model = 384
n_layer = 6 
n_head = 6
d_q = int(d_model / n_head) 
dropout = 0.2
vocab_size = 65

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

"""
---- Device ----
"""

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA (GPU)")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device('cpu')
    print("Using device's CPU")

"""
--- WandB Integration ---
"""


wandb.init(
    project="nano-model-shakesphere-training",
    config={
        "learning_rate": learning_rate,
        "architecture": "decoder-only-model",
        "dataset": "tinyshakesphere",
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": n_head,
        "max_iters": max_iters,
        "dropout": dropout
    }
)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) # --> All unique characters within the text 
vocab_size = len(chars) # 65 different characters in text

stoi = {}
itos = {}

for i in range(len(chars)):
    stoi[chars[i]] = i  # Convert strings to ints
    itos[i] = chars[i]  # Convert ints to strings

# Take a string, and output its characters indices in a list
def encoder(s):
    res = []
    for char in s:
        res.append(stoi[char])
    return res

# Take a list of indices and output a string
def decoder(l):
    res = ""
    for i in l:
        res += itos[i]
    return res

data = torch.tensor(encoder(text), dtype=torch.long) # --> Same shape as length, i.e., number of characters

n = int(0.9 * len(data))
train_data = data[:n] # 90% of text
val_data = data[n:]  # 10% of text

def get_batch(split):
    if split.lower() == 'train':
        data = train_data
    else:
        data = val_data

    ix = torch.randint(len(data)-max_len, (batch_size,)) # Generate batch_size=64 random numbers from 0 to len(data)-max_len

    x = torch.stack([data[i:i+max_len] for i in ix])        # Generates 250 ids from that random number and stacks batch_size by rows, so shape[64, 256]
    y = torch.stack([data[i+1:i+max_len+1] for i in ix])    # This is done in order to test teh real y with the later predicted y by the model using cross entropy and update weights
    
    return x.to(device), y.to(device)

"""
--- Model Training ---
"""

if __name__ == "__main__":

    from model import Model

    model = Model()
    m = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    @torch.no_grad
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            wandb.log({
                "iter": iter,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": learning_rate
            })
        iter_start = time.time()
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) # Required for new resetting as after iter, new set of batches will come
        loss.backward()                       # Required for back passing, it gives you the amount of steepness and gradient
        optimizer.step()                      # Required for actually nudging in that given direction (Taking a plausible value of lr right now but it influences a lot)

        iter_time = time.time() - iter_start
        print(f"Iteration {iter} completed in {iter_time:.2f} seconds")
        wandb.log({"iter_time": iter_time})

    wandb.finish()

    print("Training finished. Saving model state...")
    torch.save(model.state_dict(), 'nanogpt_model.pth')
    print("Model saved to nanogpt_model.pth")