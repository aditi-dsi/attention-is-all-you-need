# Attention Is All You Need - Transformers Implementation from Scratch

This project is a PyTorch-based reimplementation of the original Transformer architecture proposed in _Attention Is All You Need (Vaswani et al., 2017)_ - [Link](https://doi.org/10.48550/arXiv.1706.03762). It reconstructs the core building blocks - multi-head self-attention, positional encodings, and encoder-decoder architecture from first principles to offer interpretability and clarity.

### Structure
- `transformer_from_scratch.py` — Main script containing modular implementations of attention, feedforward layers, and positional embeddings.

- Fully built using native PyTorch (nn.Module) without relying on high-level abstractions.

### How to run
```
git clone https://github.com/aditi-dsi/attention-is-all-you-need
cd attention-is-all-you-need
python transformers_from_scratch.py
```
Requires: Python 3.8+, PyTorch 1.13+

### Acknowledgement
This implementation was developed as part of my deep learning study, and is heavily inspired by the lectures of Professor Dr. Arun Rajkumar on responsible for the course [Responsible & Safe AI Systems](https://youtu.be/mbSMzeCQ0NU?si=KJO1A9fUHMiY0mSt).


