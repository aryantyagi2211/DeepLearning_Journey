# My RNN & LSTM Journey 🧠

So this folder is basically me learning RNNs and LSTMs from scratch and trying to actually *understand* them instead of just memorizing formulas. Sharing my notes + practice here in case it helps someone else too (or future me, when I forget all this 😅).

## What clicked for me

- **RNNs read one word at a time**, not the whole sentence at once. They keep a small "memory box" (hidden state) that updates after every word — kind of like constantly rewriting a short summary in your head as you read.
- The catch: that memory box is a **fixed size**, so old stuff keeps getting mixed with new stuff and slowly fades away. It's not that the RNN "deletes" old info on purpose — it just quietly dilutes it, word after word.
- This fading thing has a scary name: the **vanishing gradient problem**. Basically the network struggles to learn from anything too far back in the sequence.
- Then comes **LSTM**, which fixes this by carrying around a second memory — the **cell state**, like a protected notebook that doesn't get squashed and rewritten every single step. It's guarded by three gates:
  - **Forget gate** – "should I erase this old info?"
  - **Input gate** – "should I write this new info in?"
  - **Output gate** – "what should I actually use right now?"
- Small detail that made it click: gates use **sigmoid** (gives a 0–1 "how much to allow" value), while the actual content being added uses **tanh**. Once I understood *why*, the whole architecture stopped feeling like magic.
- I also sat down and did an LSTM timestep completely by hand (forget → input → candidate → new cell state → output → hidden state) — genuinely helped way more than just reading about it.

## How I practiced

- Made concept notes with my own diagrams to explain things in plain English — `RNN_Journey.pdf`
- Solved a set of 50 practice questions mixing theory + hand calculations — `RNN_LSTM_Practice_Questions.md`
- Then checked myself against the answer key — `RNN_LSTM_Practice_Questions_with_Answers.md`

## Files here

| File | What's inside |
|---|---|
| `RNN_Journey.pdf` | My concept notes + diagrams |
| `RNN_LSTM_Practice_Questions.md` | 50 practice questions (try them yourself first!) |
| `RNN_LSTM_Practice_Questions_with_Answers.md` | Same questions, fully answered |

---
Part of my ongoing Deep Learning roadmap: **ANN → CNN → RNN → NLP → Transformer → LLM → Fine-tuning** — slowly but surely getting there 🚀
