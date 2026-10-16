# 💬 Multi-Turn Dialogue Generation using Generative AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An implementation of an interactive multi-turn dialogue generation system developed as part of **Generative AI **.

The system leverages Microsoft's pretrained autoregressive language model **DialoGPT** (`microsoft/DialoGPT-medium`) to generate coherent, context-aware conversational responses in real-time.

---

## 📌 Problem Statement
**Assignment 11:** Develop applications for text generation tasks such as dialogue generation using trained Generative AI models.

---

## ⚙️ Architecture & Workflow

```text
User Input ──► BPE Tokenizer ──► Token IDs + [EOS]
                                      │
           Previous Chat History ◄────┴────► Tensor Concatenation
                                                    │
                                                    ▼
                                           DialoGPT-medium
                                     (Autoregressive Generation)
                                                    │
                                                    ▼
                                            Decoding Strategy:
                                     - Nucleus Sampling (top_p = 0.95)
                                     - Top-k (k = 50)
                                     - Temperature (0.75)
                                     - Repetition Penalty (no_repeat_ngram = 3)
                                                    │
                                                    ▼
                                           Decoded Text Output
