# src/models/bert_embeddings.py
import numpy as np
import os

def load_bert_embeddings(model_dir="../models"):
    path = os.path.join(model_dir, "bert_embeddings.npy")
    if os.path.exists(path):
        print(f"Loaded BERT embeddings from {path}")
        return np.load(path)
    else:
        raise FileNotFoundError(f"Cannot find {path}")
    
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load model + tokenizer
MODEL_NAME = "bert-base-uncased"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME).to(_device)
_model.eval()

@torch.no_grad()
def embed_texts(texts, max_len=64, batch_size=16):
    """Trả về np.ndarray (n_samples, 768) CLS embeddings."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = _tokenizer(batch, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt").to(_device)
        out = _model(**enc)
        cls_vecs = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_vecs.append(cls_vecs)
    return np.vstack(all_vecs)

# alias
get_embeddings = embed_texts

