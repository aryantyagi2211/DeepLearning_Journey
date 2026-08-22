# Artificial Neural Networks (ANN) — My Learning Journey

Documenting how I learned ANN from scratch — concepts first, then hand calculations, then code, then practice questions. No shortcuts, no jumping straight to frameworks.

---

## My Learning Philosophy

```
Concept -> Hand Calculation -> Code -> Questions
```

I learn the idea in plain language first, then verify it by calculating numbers by hand, then translate that exact calculation into code (raw NumPy first, then frameworks), and finally test myself with practice questions before moving on.

---

## What I Did

1. **Revised ANN theory** — architecture, forward pass, activation functions, loss, backprop, gradient descent.
2. **Hand-calculated a small network** — 1 hidden layer, 2 neurons, sigmoid + BCE loss. Forward pass, loss, every gradient, one full weight update — all by hand first.
3. **Coded it in raw NumPy** — matched the hand-calculated numbers exactly, using descriptive variable names (not `dW1`, but `gradient_w_hidden_1`).
4. **Rebuilt in TensorFlow** — using `tf.Variable` and `GradientTape()`, matched the same gradients as the manual calculation.
5. **Rebuilt in Keras** — `model.fit()` — by this point I knew exactly what was happening inside that one line.
6. **Real training — Breast Cancer dataset** (569 patients, 30 features). Full pipeline: load → split → scale → build → compile → train → evaluate → predict.
7. **Compared this pipeline to CNN, Transfer Learning, and object detection (Faster R-CNN, SSD, YOLOv8)** — what stays the same, what changes.

---

## Files in This Folder

| File | What it is |
|---|---|
| `ANN_Notes.pdf` | Full concept notes, with diagrams |
| `ANN_Training_on_Real_Data_(Breast_Cancer).ipynb` | Real training pipeline on the Breast Cancer dataset |
| `ann_50_practice_questions.md` | 50 practice questions (concept, math, pipeline, scenarios) |
| `ann_50_questions_answer_key.md` | Answer key for self-checking |
| `neural_network.ipynb` | Manual/from-scratch neural network build |
| `ANN_Math.pdf` | Hand-calculation walkthrough on a real Breast Cancer patient sample *(adding soon)* |

---

## Biggest Doubts I Cleared Up

- **Input neurons = features, not samples.** Same network architecture reused across every sample.
- **z vs a**: `z` = before activation, `a` = after. Always compute both separately.
- **Backprop ≠ gradient descent.** Backprop only computes gradients; gradient descent applies the update.
- **fit_transform (train only) vs transform (test/new data only)** — mixing these up leaks test data into training.
- **random_state (reproducible shuffle) vs stratify (balanced class split)** — two different jobs, not the same thing.

---

## What's Next

```
ANN (done) -> CNN -> Object Detection -> Transformer -> RNN -> NLP -> LLM -> Fine-tuning
```

Parallel track: Agentic AI (CrewAI + Pipecat). CNN theory/math/detection concepts already covered — next up is hands-on CNN training, then YOLOv8 + transfer learning.
