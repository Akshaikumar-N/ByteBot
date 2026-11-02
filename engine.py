from pathlib import Path

import torch
from tokenizers import Tokenizer
from architecture import GPT

CHECKPOINT_PATH = Path("/kaggle/input/datasets/akshaikumarn/bytebot/checkpoints/last.pt")
TOKENIZER_PATH = Path("/kaggle/input/datasets/akshaikumarn/bytebot-tokenizer/tokenizer.json")

_model = None
_tokenizer = None
_device = None
_eos_id = None

def _load():
    global _model, _tokenizer, _device, _eos_id
    
    if _model is not None:
        return

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    _model = GPT(
        vocab_size=34_048,
        d_model=768,
        n_layers=14,
        n_heads=12,
        ffn_dim=2048,
        context_length=1024,
    )

    _model.load_state_dict(checkpoint["model"])
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device).eval()
    _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    _eos_id = _tokenizer.token_to_id("<eos>")
    del checkpoint

@torch.inference_mode()
def ask(prompt, max_new_tokens=150, temperature=0.8, top_k=50):
    _load()

    ids = torch.tensor(
        [_tokenizer.encode(prompt).ids],
        dtype=torch.long,
        device=_device,
    )

    if ids.shape[1] == 0:
        raise ValueError("Prompt must contain at least one token.")

    for _ in range(max_new_tokens):
        logits, _ = _model(ids[:, -1024:])
        logits = logits[:, -1, :].float()

        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)

        if next_id.item() == _eos_id:
            break
        ids = torch.cat((ids, next_id), dim=1)
    print(_tokenizer.decode(ids[0].tolist()))