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
Covered all nine subtopics in depth, building on the Transformer foundation:
- Tokenization (BPE) — built a from-scratch Byte Pair Encoding tokenizer
- Embeddings + RoPE (Rotary Positional Embeddings)
- Architecture Variants — RMSNorm, GQA (Grouped Query Attention), SwiGLU
- BERT vs GPT contrast — encoder-only vs decoder-only, and why it matters
- Pretraining — built `mini_gpt.py`, a full GPT-2-style decoder-only transformer (tiktoken, causal self-attention, AdamW, autoregressive generation), plus a pretraining demo
- Scaling Laws
- Inference & KV-Cache
- Context Length
- MoE (Mixture of Experts)

Also put together a 17-page `LLM_Journey.pdf` with notes and diagrams for the whole topic.

---

## 🔜 What's next

- **Fine-tuning** — SFT, LoRA, QLoRA, RLHF/DPO
- After that: building both a **GPT-style** and a **Llama-style** model completely from scratch, using everything from the LLM topic

Alongside this, I'm also building a parallel **Agentic AI curriculum** focused on **CrewAI** and **Pipecat**, working toward building real agentic systems once the core DL foundation is solid.

---

## 📁 How this repo is structured

Each topic gets its own folder with:
- Notes (concept + math, hand-calculated where it matters)
- Code (raw NumPy implementation first, then PyTorch/TensorFlow)
- A topic-level README

I also post my hand-calculation notes and progress on [LinkedIn](#) as I go, and commit here daily.

---

## 📝 Sprint Notes

At the end of every major topic, I put together a **sprint note** — a compact, revision-focused PDF that pulls together every concept from that topic into one place: diagrams first, then short bullet-point explanations, callout boxes for common interview traps/gotchas, and a quick-recall Q&A section at the end.

The idea is simple: by the time I finish a topic, the raw notes/code are scattered across many days of work — the sprint note compresses all of that into something I can re-read in 20-30 minutes before an interview or before starting the next topic, instead of digging back through weeks of daily notes.

I've also built a full **DL Interview & MNC-Project Master Guide** — one consolidated sprint note covering ANN → CNN → RNN → NLP → Transformer → LLM together, with a rapid-fire cheat-sheet table at the end for last-minute revision.

## 🎯 Practicing with 100 Scenario-Based Interview Questions

To actually test whether I understand this stuff (and not just recognize it when I see it), I built a **100-question scenario bank** styled after real MNC interview rounds (Google, Microsoft, NASA, Tesla, Nvidia, Amazon flavored problems) — covering model selection, CNN/vision, sequence models, NLP, Transformers, LLMs, and system design/MLOps.

Every question is a real-world situation, not a definition — e.g. *"Tesla needs a model to run in under 10ms on embedded hardware with no cloud connection — what do you choose and why?"* — instead of *"What is quantization?"*

How I practice with it:
1. Go section by section, one scenario at a time.
2. Read only the scenario + a one-line hint — no answers visible yet.
3. Actually answer out loud or write it down, like a real interview, before looking anything up.
4. Only then check the answer key (kept in a separate section at the back of the PDF) and compare my reasoning against the ideal approach.

This forces me to practice *decision-making under constraints* (latency, data size, cost, safety) instead of just recalling facts — which is what real interviews actually test.

---

Follow along, and feel free to open issues if you spot mistakes in my notes — I'm learning in public, mistakes and all. 🙂