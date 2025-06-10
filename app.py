import torch
import gradio as gr

from model import Model
from train import encoder, decoder

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Model().to(device)
model.load_state_dict(torch.load("nanogpt_model.pth", map_location=device))
model.eval()

# Generation function
def generate_text(prompt, max_tokens):
    idx = torch.tensor(encoder(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(idx, max_new_tokens=max_tokens)[0].tolist()
    return decoder(generated)

# Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a prompt...", label="Prompt"),
        gr.Slider(10, 500, value=200, step=10, label="Max Tokens")
    ],
    outputs=gr.Textbox(label="Generated Output"),
    title="🧠 NanoGPT from Scratch",
    description="A tiny GPT model trained on Shakespeare. Try your luck by giving it a prompt!"
)

iface.launch()