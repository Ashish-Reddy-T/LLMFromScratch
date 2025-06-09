import torch

batch_size = 64     # Batch size
max_len = 256       # Maximum positional embeddings --> Position Matrix with shape (256, d_model)

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
    
    return x, y

if __name__ == "__main__":
    xb, yb = get_batch('train')
    print("\n--- Example Batch ---")
    print("Input batch shape:", xb.shape)
    print("Target batch shape:", yb.shape)

    print("\nExample of one sequence in the batch:")
    print("Input (x):", xb[0])
    print("Target (y):", yb[0])

    print("\nLet's see what this looks like as text:")
    for i in range(max_len):
        context = xb[0][:i+1]
        target = yb[0][i]
        print(f"When input is '{decoder(context.tolist())}', the target is '{decoder([target.tolist()])}'")