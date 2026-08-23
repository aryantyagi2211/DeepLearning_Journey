# Transfer Learning — 50 Practice Questions

Covers everything from this session: architecture (Phase 1 & Phase 2), math/gradients, callbacks, data augmentation, the full fine-tuning pipeline, and scenario-based reasoning.

Try answering these yourself first — no answer key given on purpose, so you actually think them through. We can go over your answers together after.

---

## Section A — Architecture & Basic Concepts (Q1–10)

1. What is transfer learning and how is it different from training a model from scratch?
2. What is the difference between feature extraction and fine-tuning in transfer learning?
3. Why do we freeze the base model in Phase 1?
4. What is catastrophic forgetting?
5. What factors do you consider when choosing a pretrained model?
6. What kind of tasks benefit from ImageNet pretrained weights?
7. Is transfer learning useful when the target domain is very different from the source domain (e.g. natural images vs medical X-rays)?
8. Why do we use Global Average Pooling instead of Flatten before the dense head?
9. What does `include_top=False` mean when loading a pretrained model?
10. In what scenarios does transfer learning not work well or should be avoided?

## Section B — Math & Technical Concepts (Q11–20)

11. If a layer is frozen, does the forward pass still happen through it? Explain using the chain rule.
12. How does gradient flow through frozen layers even though their weights don't update?
13. Why is a very small learning rate (like `1e-5`) used in Phase 2? What is the mathematical effect on the weight update?
14. Why must BatchNorm layers be kept in inference mode (`training=False`) even when frozen?
15. Write the formula for Global Average Pooling and compute a small numerical example.
16. What role does weight decay/regularization play during fine-tuning?
17. If you unfreeze the entire base model and train with a normal learning rate (0.001), what happens to the gradients and why is it a problem?
18. When would you use softmax vs sigmoid in the output layer of the dense head?
19. Write the categorical cross-entropy loss formula and explain each symbol.
20. How do you read a training vs validation loss curve to detect overfitting?

## Section C — Training Dynamics, Callbacks & Augmentation (Q21–30)

21. What does `base_model.trainable = False` actually do internally?
22. Why do you need to recompile the model after changing trainable flags in Phase 2?
23. What is the purpose of the EarlyStopping callback? What does `patience` control?
24. What does `restore_best_weights=True` do in EarlyStopping?
25. What is the purpose of the ModelCheckpoint callback?
26. Why is data augmentation important for transfer learning specifically?
27. Why does a data augmentation layer behave differently during training vs inference?
28. What happens to the dense head's weights if you skip Phase 1 and go straight to Phase 2?
29. How can `model.summary()` help you verify how many parameters are trainable vs frozen?
30. What is the `initial_epoch` parameter used for when continuing from Phase 1 to Phase 2?

## Section D — Code / Pipeline (Q31–40)

31. Write the code to load a pretrained MobileNetV2 model without its top classification layer.
32. Write the code to freeze the entire base model.
33. Write the code to build a new model with `GlobalAveragePooling2D` and a `Dense` output head on top of the frozen base.
34. Write the code to unfreeze only the last 20 layers of a pretrained model while keeping earlier layers frozen.
35. Write the `compile()` call for Phase 2 with an appropriately small learning rate.
36. Write the `fit()` call for Phase 2 that continues epoch counting from where Phase 1 left off.
37. Write the code to add EarlyStopping and ModelCheckpoint callbacks and pass them into `fit()`.
38. Write a data augmentation block using `RandomFlip`, `RandomRotation`, and `RandomZoom`.
39. Write a small function that prints how many layers in a model are trainable vs frozen.
40. List, in order, the full pipeline of steps used to go from a pretrained model to a fully fine-tuned model.

## Section E — Scenario-Based Questions (Q41–50)

41. **Scenario:** You have only 300 images across 4 classes. Would you go straight to fine-tuning the whole network, or start with Phase 1? Why?

42. **Scenario:** After unfreezing the base model in Phase 2, your validation accuracy suddenly drops sharply. What's the most likely cause and how would you fix it?

43. **Scenario:** You forget to recompile the model after changing trainable flags. What will actually happen when you call `fit()` again?

44. **Scenario:** Your model overfits badly even with transfer learning and a small dataset. What are two things you could try, based on what we covered?

45. **Scenario:** A teammate unfreezes the entire base model right after loading pretrained weights, skipping Phase 1 completely, and uses a learning rate of 0.001. What will likely go wrong?

46. **Scenario:** You're transferring a model pretrained on natural images to X-ray classification. What extra concerns should you think about compared to a natural-image target task?

47. **Scenario:** Your training accuracy is going up but validation accuracy plateaus early in Phase 2. What callback would help you avoid wasting further epochs, and how would you configure it?

48. **Scenario:** You want to save the best-performing model automatically during Phase 2 training, in case fine-tuning starts to hurt performance later. What would you set up before calling `fit()`?

49. **Scenario:** You're asked in an interview why transfer learning works at all — why can features learned on ImageNet (cats, dogs, cars) help with a completely different task like classifying satellite images? How do you explain this?

50. **Scenario:** Your manager asks why you use a 100x smaller learning rate in Phase 2 compared to Phase 1. Explain this in simple terms as if to a non-technical person.

---

Try answering as many as you can. We'll go through your answers together and correct/clarify anything needed before moving to the next topic.
