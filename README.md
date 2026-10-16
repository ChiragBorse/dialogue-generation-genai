# 💬 Multi-Turn Dialogue Generation using Generative AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, multi-turn conversational AI system developed using pretrained autoregressive Transformer models. This project demonstrates how causal language models manage conversation state, maintain context memory, and decode natural language responses in real time.

---

## 📌 Problem Statement
**Objective:** Develop applications for text generation tasks such as dialogue generation using trained Generative AI models.

---

## ❓ Project Overview (The 5 Ws)

### 1. What is this project?
An end-to-end multi-turn dialogue agent built on top of Hugging Face Transformers and PyTorch. Instead of basic single-turn Q&A, it concatenates conversational turns to maintain memory across a multi-turn conversation and uses nucleus and top-$k$ sampling to generate context-aware, human-like replies.

### 2. Why build it?
* **Beyond Rule-Based Systems:** Traditional chatbots rely on rigid if-else statements or regex patterns. Generative models synthesize unique responses dynamically based on natural language context.
* **Context Preservation:** Dialogue requires remembering prior turns. This project demonstrates how sequential tensor concatenation passes multi-turn state directly into causal attention masks.
* **Text Decoding Control:** It shows how hyperparameter tuning (temperature, top-$k$, top-$p$, repetition penalty) directly mitigates common generative issues like hallucinations and repetitive loops.

### 3. Who is this for?
* **Students & Researchers:** Understanding causal language modeling, decoder architectures, and NLP decoding strategies.
* **AI Developers:** A reference template for building lightweight conversational CLI tools or backends for web bots.

### 4. Where can it be run?
* **Google Colab (Recommended):** Best for instant setup without local hardware requirements; runs smoothly on a free T4 GPU or standard CPU.
* **Local Machine / Lab Terminal:** Compatible with Windows, macOS, or Linux using standard Python virtual environments.

### 5. How does it work?
The system loads a pretrained checkpoint of Microsoft's `DialoGPT-medium` (a GPT-2 variant fine-tuned on 147 million Reddit conversational threads). User prompts are encoded using Byte-Pair Encoding (BPE), appended to previous conversation token IDs, passed through causal self-attention layers, and decoded into readable strings.

---

## ⚙️ Architecture & Technical Workflow

```text
 User Input ("Hi!")
        │
        ▼
 ┌───────────────────────────┐
 │   BPE Tokenizer (GPT-2)   │ ──► Produces input token IDs + [EOS] token
 └───────────────────────────┘
        │
        ▼
 ┌───────────────────────────┐
 │   Tensor Concatenation    │ ◄── Appends new tokens to previous conversation history
 └───────────────────────────┘
        │
        ▼
 ┌───────────────────────────┐
 │   DialoGPT-medium Model   │ ──► Autoregressive generation via causal self-attention
 └───────────────────────────┘
        │
        ▼
 ┌───────────────────────────┐
 │     Decoding Strategy     │
 │  • Top-k: 50              │ ──► Narrows candidate vocabulary to top 50 likely tokens
 │  • Nucleus (Top-p): 0.95  │ ──► Caps cumulative probability distribution at 95%
 │  • Temperature: 0.75      │ ──► Flattens distribution slightly for natural creativity
 │  • No-Repeat N-Gram: 3    │ ──► Prevents 3-word repetitive phrase loops
 └───────────────────────────┘
        │
        ▼
 ┌───────────────────────────┐
 │   Detokenizer & Cleanup   │ ──► Strips historical prompt tokens & formats response
 └───────────────────────────┘
        │
        ▼
 Bot Response ("Hello! How can I help you today?")
