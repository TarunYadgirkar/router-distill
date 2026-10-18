"""Mean-pooled sentence embeddings from all-MiniLM-L6-v2, read from the local Hugging Face cache.
Needs the optional `embed` dependency group (torch, transformers)."""

import os

import numpy as np

# Read the weights from the local cache only; set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 64


def embed(texts: list[str]) -> np.ndarray:
    """(len(texts), 384) unit-norm embeddings; nothing is downloaded when the model is cached."""
    import torch
    import transformers

    torch.manual_seed(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ENCODER)
    model = transformers.AutoModel.from_pretrained(ENCODER).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = tokenizer(
                texts[start : start + BATCH], padding=True, truncation=True, return_tensors="pt"
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            chunks.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    return np.vstack(chunks)
