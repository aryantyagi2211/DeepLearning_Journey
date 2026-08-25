# Transfer Learning — My Journey

This folder is part of my Deep Learning journey (CNN → Transfer Learning).
Here I learned how to reuse an already-trained model instead of training one from scratch.

## What I learned

- A pretrained model (like MobileNetV2, VGG16) already knows how to detect generic stuff — edges, textures, shapes — because it was trained on millions of images (ImageNet).
- Instead of training a new CNN from zero, I can reuse that knowledge and just teach it my own classes.
- This is way faster and works even with a small dataset.

## Two phases I practiced

**Phase 1 — Feature Extraction**
- Freeze the whole pretrained base model (its weights don't change)
- Add a new head (GlobalAveragePooling + Dense layer) on top
- Train only this new head

**Phase 2 — Fine-Tuning**
- Unfreeze the last few layers of the base model
- Train with a very small learning rate (like `1e-5`) so I don't destroy what the model already knows
- This helps the model adjust a bit more specifically to my dataset

## Why do Phase 1 before Phase 2?

If I unfreeze everything right away, the new head (which starts random) sends big error signals into the pretrained model and messes up its good weights. Doing Phase 1 first lets the head stabilize, so Phase 2 fine-tuning is safe and gentle.

## Extra things I practiced

- **Callbacks** — EarlyStopping (stop training when it's not improving) and ModelCheckpoint (auto-save the best model)
- **Data Augmentation** — flipping/rotating/zooming images randomly during training so the model doesn't overfit on a small dataset

## Files in this folder

| File | What it is |
|---|---|
| `MobileNetV2_Flower_Classification_Explained-Copy1.ipynb` | My first transfer learning practice using MobileNetV2 |
| `Transfer_Learning_Concept_Notes.docx` | Concept notes with diagrams (Phase 1 & Phase 2) |
| `Transfer_Learning_50_Questions.md` | 50 practice questions (self-test) |
| `Transfer_Learning_50_Questions_Answers.md` | Same questions with answers |
| `Transfer_Learning_VGG16.ipynb` | Same Phase 1 + Phase 2 pipeline, done again using VGG16 |

## Models I practiced with

- MobileNetV2
- VGG16

## Next up

RNN → NLP → LLM → Fine-tuning
