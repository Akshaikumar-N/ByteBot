# ByteBot

**ByteBot** is a decoder-only language model built from scratch in
PyTorch. This repository documents the data preparation, tokenizer
training, model architecture, pretraining, retraining, and inference
workflow. A public Kaggle inference notebook is provided so others can
load the model and experiment with text generation.

> ByteBot is an experimental, base language model. It is not an
> instruction-tuned assistant, and its responses may be incomplete,
> inaccurate, or incoherent.

## Try ByteBot

[Open the public Kaggle inference
notebook](https://www.kaggle.com/code/akshaikumarn/inference/edit)

The notebook imports `ask` from `engine.py` and runs a sample prompt.
You can edit the prompt and generation settings to experiment.

## Project workflow

The project was developed in stages:

1.  **Corpus preparation** -- source datasets were downloaded/streamed
    from Hugging Face, shuffled, filtered, and combined into text files.
2.  **Tokenizer training** -- a byte-level BPE tokenizer was trained on
    a sample of the prepared corpus and saved as `tokenizer.json`.
3.  **Tokenization and binary data creation** -- the text was tokenized
    and stored as `uint16` `.bin` files, with separate training and
    validation splits.
4.  **Architecture implementation** -- the GPT model was implemented in
    `architecture.py`.
5.  **Pretraining** -- the model was trained on two NVIDIA T4 GPUs in a
    Kaggle session.
6.  **Retraining** -- training continued from the saved `last.pt`
    checkpoint in a second Kaggle session, due to Kaggle's session time
    limit.
7.  **Inference** -- `engine.py` loads the checkpoint and tokenizer and
    exposes the `ask()` function.

## Dataset and tokenizer

The corpus was assembled from five Hugging Face datasets:

  Source                Approximate target size
  ------------------- -------------------------
  FineWeb-Edu                              5 GB
  DCLM                                     3 GB
  Cosmopedia v2                            3 GB
  BookCorpus                               3 GB
  English Wikipedia                        2 GB
  **Total target**                    **16 GB**

Documents were streamed and shuffled using a buffer, filtered to remove
very short entries, and separated with an `<eos>` marker. The
tokenizer-training step sampled approximately 25% of the prepared
documents.

### Tokenizer

-   Algorithm: Byte-level Byte-Pair Encoding (BPE)
-   Vocabulary size: 34,048
-   Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
-   Saved artifact: `tokenizer.json`

### Binary token files

The tokenized corpus was split into training and validation data. Each
source produced one binary file in each split:

``` text
bytebot_train/
├── fineweb_edu.bin
├── dclm.bin
├── cosmopedia_v2.bin
├── bookcorpus.bin
└── wikipedia.bin

bytebot_val/
├── fineweb_edu.bin
├── dclm.bin
├── cosmopedia_v2.bin
├── bookcorpus.bin
└── wikipedia.bin

bytebot-tokenizer/
└── tokenizer.json
```

The validation split was configured to receive approximately 0.5% of
documents. Binary token IDs are stored as `uint16` and read using NumPy
memory mapping during training.

## Model architecture

ByteBot uses a GPT-style causal Transformer decoder.

  Configuration                           Value
  ------------------------ --------------------
  Parameters                 Approximately 125M
  Vocabulary size                        34,048
  Model dimension                           768
  Transformer layers                         14
  Attention heads                            12
  Feed-forward dimension                  2,048
  Context length                   1,024 tokens

Architecture components include:

-   Causal self-attention using PyTorch scaled dot-product attention
-   Rotary positional embeddings (RoPE)
-   RMSNorm
-   SwiGLU-style feed-forward layers
-   Residual connections with residual scaling
-   Tied token-embedding and language-model-head weights

The implementation is in `architecture.py`.

## Training

Training was performed on Kaggle using **two NVIDIA T4 GPUs** and
mixed-precision FP16.

The training setup includes:

-   AdamW optimizer
-   Gradient accumulation
-   Gradient clipping
-   Linear warm-up followed by cosine learning-rate decay
-   Periodic validation and checkpoint saving
-   Random sampling of fixed-length token sequences from memory-mapped
    `.bin` files

The initial training run used a learning rate of `3e-4` with a minimum
learning rate of `3e-5`. It ran for approximately 10 hours.

A second run resumed from the saved `last.pt` checkpoint, using a
learning rate of `1e-4` and a minimum learning rate of `1e-5`, for
approximately 11 hours. The continuation was necessary because an
individual Kaggle GPU session has a time limit.

The training notebook saves checkpoints such as:

-   `last.pt` -- latest resumable checkpoint, including model and
    optimizer/training state
-   `best.pt` -- checkpoint saved when validation loss improves

The retraining notebook references the earlier checkpoint at:

``` python
source_checkpoint = Path(
    "/kaggle/input/notebooks/akshaikumarn/bytebot-trainer/checkpoints/last.pt"
)
```

The exact storage location can differ depending on how the Kaggle
notebook output is attached as an input dataset.

## Repository files

``` text
.
├── architecture.py
├── engine.py
├── train.ipynb
├── corpus-tokenizer_build.ipynb
├── inference.ipynb
├── README.md
└── tokenizer.json        # distribute separately if not committed
```

-   `architecture.py` -- model layers and GPT architecture.
-   `corpus-tokenizer_build.ipynb` -- corpus streaming, tokenizer
    training, and `.bin` generation.
-   `train.ipynb` -- model pretraining and checkpoint continuation
    workflow.
-   `engine.py` -- checkpoint/tokenizer loading and autoregressive
    generation.
-   `inference.ipynb` -- minimal example that calls the inference
    engine.

Large artifacts (the `.bin` files and model checkpoints) are stored
separately in Kaggle datasets rather than being committed to this Git
repository.

## Running inference

The inference engine expects the model checkpoint and tokenizer at the
paths configured near the top of `engine.py`:

``` python
CHECKPOINT_PATH = Path(
    "/kaggle/input/datasets/akshaikumarn/bytebot/checkpoints/last.pt"
)
TOKENIZER_PATH = Path(
    "/kaggle/input/datasets/akshaikumarn/bytebot-tokenizer/tokenizer.json"
)
```

Attach the corresponding Kaggle datasets to your notebook, or update
these paths to match your own storage location.

Install the required packages in your environment:

``` bash
pip install torch tokenizers
```

Keep `architecture.py`, `engine.py`, the checkpoint, and
`tokenizer.json` accessible to the notebook. Then:

``` python
from engine import ask

ask(
    "Machine learning is a branch of artificial intelligence that",
    max_new_tokens=150,
    temperature=0.8,
    top_k=50,
)
```

`ask()` prints the generated text.

### Generation arguments

  -----------------------------------------------------------------------
  Argument                            Meaning
  ----------------------------------- -----------------------------------
  `prompt`                            Text used to start generation

  `max_new_tokens`                    Maximum number of new tokens to
                                      generate

  `temperature`                       Sampling randomness; values at or
                                      below 0 use greedy decoding

  `top_k`                             Restricts sampling to the
                                      highest-scoring `k` tokens; set to
                                      `None` or a non-positive value to
                                      disable top-k filtering
  -----------------------------------------------------------------------

The model uses a maximum context length of 1,024 tokens. During
generation, the engine feeds the most recent 1,024 tokens to the model.

## Important notes

-   The model checkpoint and tokenizer must match. Use the tokenizer
    trained for this model.
-   The inference engine currently uses fixed architecture settings; if
    the architecture changes, update the configuration in `engine.py` as
    well.
-   The public Kaggle notebook may require a Kaggle account and access
    to the attached input datasets.
-   The project is shared for experimentation and learning. No claim is
    made that ByteBot matches the quality or capabilities of large
    instruction-tuned language models.

## Acknowledgements

The corpus uses publicly available datasets hosted on Hugging Face.
Please refer to each dataset's own page for its license, terms of use,
and attribution requirements before redistributing any derived data.
