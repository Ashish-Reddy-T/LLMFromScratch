import torch
from model import Model
from train import encoder, decoder

"""
--- Device ---
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


if __name__ == "__main__":
    model = Model()
    m = model.to(device)

    print("Loading model parameters...")
    m.load_state_dict(torch.load('nanogpt_model.pth', map_location=device))

    m.eval()
    print("Model loaded successfully.")

    print("\n\n--- Generating new text ---")
    context = torch.tensor(encoder('\n'), dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = m.generate(context, max_new_tokens=1000)[0].tolist()
    print(decoder(generated_tokens))