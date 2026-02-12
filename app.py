import os
# --- FIX: This must be set BEFORE importing torch ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#C:/Users/HP/miniconda3/envs/env_ML/python.exe -m streamlit run c:/Users/HP/Desktop/Desktop/GENAI/ASS1/app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tokenizers import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(2048, hidden_size)
    def forward(self, x):
        x = self.fc(x)
        return x.unsqueeze(0)

class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, captions, hidden):
        emb = self.embed(captions).permute(1, 0, 2)
        out, hidden = self.gru(emb, hidden)
        preds = self.fc(out) 
        return preds, hidden

class CaptionModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size, vocab_size)
    def forward(self, img_fea, captions):
        hidden = self.encoder(img_fea)
        outputs, _ = self.decoder(captions, hidden)
        return outputs

@st.cache_resource
def load_resources():
    tokenizer = Tokenizer.from_file("tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    model = CaptionModel(hidden_size=512, vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load("caption_model.pth", map_location=device))
    model.eval()

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    
    return tokenizer, model, resnet, transform

def generate_caption(image, tokenizer, model, resnet, transform):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = resnet(img_tensor).view(-1)

    model.eval()
    img_feat = img_feat.unsqueeze(0).to(device)
    hidden = model.encoder(img_feat)
    start_id = tokenizer.token_to_id("<start>")
    end_id   = tokenizer.token_to_id("<end>")
    
    beams = [([start_id], 0.0, hidden)]
    for _ in range(30): # max_len
        new_beams = []
        for seq, score, hid in beams:
            if seq[-1] == end_id:
                new_beams.append((seq, score, hid))
                continue
            last_word = torch.tensor([[seq[-1]]]).to(device)
            out, new_hid = model.decoder(last_word, hid)
            probs = F.log_softmax(out[0,0], dim=0)
            topk_probs, topk_ids = torch.topk(probs, 5) # beam_width
            for i in range(5):
                next_id = topk_ids[i].item()
                next_score = score + topk_probs[i].item()
                new_beams.append((seq + [next_id], next_score, new_hid))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:5]
    
    best_seq = beams[0][0]
    return tokenizer.decode(best_seq, skip_special_tokens=True)

st.title(" Image Caption Generator")

try:
    tokenizer, model, resnet, transform = load_resources()
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        
        if st.button("Generate Caption"):
            with st.spinner("Thinking..."):
                caption = generate_caption(image, tokenizer, model, resnet, transform)
                st.success(f"**Caption:** {caption}")

except Exception as e:
    st.error(f"Error loading model or resources: {e}")