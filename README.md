# Deep Learning Journey 🚀

Hey! I'm Ayan, and this repo is where I document my self-taught Deep Learning journey — not just the theory, but the concept, the math (hand calculations), and the code for every topic I learn.

I'm following a structured roadmap:

**ANN → CNN → RNN → NLP → Transformer → LLM → Fine-tuning**

No skipping around, no passive video-watching — I build everything from scratch first (raw Python/NumPy), then implement it properly in PyTorch/TensorFlow, and I make sure I can do the hand calculations before I touch any code.

---

## ✅ What I've covered so far

### 1. ANN (Artificial Neural Networks)
Started from the basics — perceptrons, forward pass, activation functions, loss functions, and backpropagation by hand before ever writing `model.fit()`. This gave me the foundation for everything that came after.

### 2. CNN (Convolutional Neural Networks)
Went deep into:
- Convolution layers, padding, stride, pooling — and how backprop actually flows through a conv layer (not just the forward pass)
- Famous CNN architectures
- Transfer learning
- **Object Detection**: R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN) and SSD (Single Shot Detector) — how region proposals work, ROI pooling, anchor boxes, and how detection differs from plain classification

### 3. RNN & LSTM
Covered the full architecture in depth:
- What RNN Actually Does, The Memory Box Idea (Diagram), What Is Fading?, Real Example — Fading in Action
- LSTM — The Smarter Memory (Diagram), LSTM Concept — Simple Words, LSTM Math — Worked Example by Hand

### 4. NLP
Covered the full architecture in depth:
Topic 1: Text Preprocessing
Topic 2: Text Representation (Basic)
Topic 3: Word Embeddings
Topic 4: Sequence Modeling Context
Topic 5: NLP Tasks (Classical)
Topic 6: Language Models (N-gram)
Topic 7: Evaluation Metrics

### 5. Transformer
Covered the full architecture in depth:
- Self-attention and multi-head attention (hand-calculated, not just conceptual)
- The complete Transformer pipeline worked through with real numbers, end-to-end

### 6. LLM (Large Language Models)
Covered the full pipeline, one subtopic at a time — diagram, concept, hand math, then code for each:
- **Tokenization (BPE)** — hand-ran the merge algorithm on a real sentence, counting pair frequencies myself, before writing the raw Python version
- **Embeddings + RoPE** — how token meaning gets stored, and how rotary position embedding bakes word order into Query/Key vectors through rotation instead of addition
- **Architecture variants** — RMSNorm, Grouped-Query Attention (GQA), and SwiGLU: the modern, more efficient replacements for LayerNorm, standard multi-head attention, and GELU feedforward, all hand-calculated with real numbers
- **BERT contrast** — encoder-only, bidirectional architecture, covered to understand *why* decoder-only was chosen for generation
- **Pretraining** — next-token prediction, cross-entropy loss, and the full forward → loss → backward → update loop, same mechanism as ANN, just at massive scale
- **Scaling laws** — the Chinchilla finding that model size and data size need to grow together (~20 tokens per parameter) for a given compute budget
- **Inference / KV-cache** — how real-time generation stays fast by caching Key/Value pairs instead of recomputing them every step
- **Context length** — why full attention is quadratically expensive, and the fixes (sliding window, sparse attention)
- **Mixture of Experts (MoE)** — sparse activation, how models like GPT-4 and Mixtral get huge total capacity while keeping per-token compute cheap

Built a full production-architecture decoder-only transformer (`mini_gpt.py`) — tokenization, embeddings, causal self-attention, feedforward, a complete pretraining loop, and text generation — plus a from-scratch BPE tokenizer matching my hand calculations exactly.

---

## 🔜 What's next

- **Fine-tuning** — SFT, PEFT (LoRA, QLoRA), and RLHF/DPO, adapting pretrained models for custom tasks
- Building both a GPT-style and a Llama-style model fully from scratch (using the RMSNorm/GQA/SwiGLU variants from the LLM topic) and integrating them into this project

Alongside this, I'm also building a parallel **Agentic AI curriculum** focused on **CrewAI** and **Pipecat**, working toward building real agentic systems once the core DL foundation is solid. RAG (all 10 architectures), prompt engineering, and system design are also on deck.

---

## 📁 How this repo is structured

Each topic gets its own folder with:
- Notes (concept + math, hand-calculated where it matters)
- Code (raw NumPy implementation first, then PyTorch/TensorFlow)
- A topic-level README

I also post my hand-calculation notes and progress on [LinkedIn](#) as I go, and commit here daily.

---

## Why I'm doing it this way

Anyone can copy-paste a CNN from a tutorial. My goal is to actually *understand* why each piece exists — the math behind the magic — so that when I eventually get to LLMs and fine-tuning, I'm not just calling APIs, I know what's happening underneath.

Follow along, and feel free to open issues if you spot mistakes in my notes — I'm learning in public, mistakes and all. 🙂
