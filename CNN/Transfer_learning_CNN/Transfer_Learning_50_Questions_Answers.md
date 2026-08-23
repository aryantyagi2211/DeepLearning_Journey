# Transfer Learning — 50 Practice Questions with Answers

Covers everything from this session: architecture (Phase 1 & Phase 2), math/gradients, callbacks, data augmentation, the full fine-tuning pipeline, and scenario-based reasoning.

---

## Section A — Architecture & Basic Concepts (Q1–10)

**1. What is transfer learning and how is it different from training a model from scratch?**
Transfer learning reuses a model already trained on a large dataset (e.g. ImageNet) instead of learning everything from random weights. A from-scratch model has to learn generic features like edges and textures itself, which needs huge data and time; transfer learning starts with those features already learned.

**2. What is the difference between feature extraction and fine-tuning in transfer learning?**
Feature extraction (Phase 1) freezes the entire base model and only trains a new head. Fine-tuning (Phase 2) additionally unfreezes some of the base model's later layers and trains them with a very small learning rate, alongside the head.

**3. Why do we freeze the base model in Phase 1?**
Because its pretrained weights already encode useful generic features. Freezing prevents the large, mostly-random gradients coming from the new head from disturbing this good knowledge while the head is still learning.

**4. What is catastrophic forgetting?**
When large weight updates on a pretrained model overwrite/destroy the useful features it had already learned, effectively making it "forget" what it knew before.

**5. What factors do you consider when choosing a pretrained model?**
Similarity of the source dataset to your target domain, model size vs your compute/latency budget, input resolution compatibility, and how well-documented/supported the architecture is.

**6. What kind of tasks benefit from ImageNet pretrained weights?**
Any general natural-image task — object classification, detection — since ImageNet features (edges, textures, shapes) transfer well to most everyday visual domains.

**7. Is transfer learning useful when the target domain is very different from the source domain (e.g. natural images vs medical X-rays)?**
Yes, but less directly — early layers (generic edges/textures) still transfer reasonably well, but more layers may need to be unfrozen and fine-tuned since higher-level features are less relevant.

**8. Why do we use Global Average Pooling instead of Flatten before the dense head?**
GAP averages each feature map into a single number, producing far fewer parameters than Flatten, which reduces overfitting risk and works regardless of input image size.

**9. What does `include_top=False` mean when loading a pretrained model?**
It removes the original classification head (built for the source dataset's classes, e.g. 1000 ImageNet classes), leaving only the convolutional feature-extraction layers so a new head can be attached.

**10. In what scenarios does transfer learning not work well or should be avoided?**
When the target domain is extremely different from the source (e.g. transferring from natural images to raw audio spectrograms) or when you actually have enough data and compute to train a specialized architecture from scratch that would outperform a generic pretrained one.

## Section B — Math & Technical Concepts (Q11–20)

**11. If a layer is frozen, does the forward pass still happen through it? Explain using the chain rule.**
Yes. Forward pass always runs through every layer regardless of `trainable` status, since outputs are needed for the next layer. During backpropagation, gradients still flow *through* frozen layers via the chain rule (needed to reach earlier layers), but the gradient *with respect to that layer's own weights* is simply not applied as an update.

**12. How does gradient flow through frozen layers even though their weights don't update?**
Backprop computes ∂L/∂(layer input) using the chain rule regardless of whether weights are trainable, because this is needed to keep propagating gradients backward. Only the weight-update step (`W = W - lr * dL/dW`) is skipped for frozen layers.

**13. Why is a very small learning rate (like `1e-5`) used in Phase 2? What is the mathematical effect on the weight update?**
Because `ΔW = -lr * dL/dW`. A tiny `lr` makes each update very small, so pretrained weights shift only slightly instead of being overwritten by large, potentially destabilizing updates — protecting existing knowledge while still allowing gradual adaptation.

**14. Why must BatchNorm layers be kept in inference mode (`training=False`) even when frozen?**
Because BatchNorm has two parts: learnable scale/shift parameters, and running mean/variance statistics that update during training mode regardless of `trainable`. If left in training mode, these statistics would keep changing even though the layer is "frozen," corrupting the pretrained normalization behavior.

**15. Write the formula for Global Average Pooling and compute a small numerical example.**
For a feature map of size H×W: `GAP = (1/(H*W)) * Σ(all pixel values)`. Example: a 2×2 feature map `[[1,3],[5,7]]` → GAP = (1+3+5+7)/4 = 4.

**16. What role does weight decay/regularization play during fine-tuning?**
It penalizes large weight values in the loss function, discouraging the fine-tuned weights from drifting too far from stable, generalizable values — helping prevent overfitting on a small fine-tuning dataset.

**17. If you unfreeze the entire base model and train with a normal learning rate (0.001), what happens to the gradients and why is it a problem?**
The randomly-initialized new head produces large error signals initially; combined with a normal learning rate, this pushes large gradient updates through the entire pretrained network, rapidly destroying its learned features (catastrophic forgetting).

**18. When would you use softmax vs sigmoid in the output layer of the dense head?**
Softmax for multi-class, single-label classification (probabilities sum to 1 across classes). Sigmoid for binary classification or multi-label classification (each class probability is independent).

**19. Write the categorical cross-entropy loss formula and explain each symbol.**
`L = -Σ y_i * log(ŷ_i)`, where `y_i` is the true one-hot label (1 for the correct class, 0 otherwise) and `ŷ_i` is the predicted probability for class `i`. The sum runs over all classes.

**20. How do you read a training vs validation loss curve to detect overfitting?**
If training loss keeps decreasing but validation loss flattens or starts increasing, the model is overfitting — memorizing training data instead of generalizing.

## Section C — Training Dynamics, Callbacks & Augmentation (Q21–30)

**21. What does `base_model.trainable = False` actually do internally?**
It sets the `trainable` attribute to `False` on every weight/variable inside that model, so the optimizer skips computing weight updates for them during `fit()`, even though forward/backward signal still passes through.

**22. Why do you need to recompile the model after changing trainable flags in Phase 2?**
Because Keras "freezes" the set of trainable variables at compile time for the optimizer; changing `trainable` flags afterward requires recompiling so the optimizer picks up the new list of variables to update.

**23. What is the purpose of the EarlyStopping callback? What does `patience` control?**
It stops training automatically once a monitored metric (e.g. `val_loss`) stops improving, avoiding wasted epochs and overfitting. `patience` sets how many consecutive epochs without improvement are tolerated before stopping.

**24. What does `restore_best_weights=True` do in EarlyStopping?**
When training stops, it rolls the model back to the weights from the epoch with the best monitored metric, rather than keeping the weights from the final (possibly worse) epoch.

**25. What is the purpose of the ModelCheckpoint callback?**
It saves the model to disk during training — typically only when the monitored metric improves — so the best version is preserved even if training later degrades or crashes.

**26. Why is data augmentation important for transfer learning specifically?**
Transfer learning is often used with small datasets, which are prone to overfitting. Augmentation artificially increases variety (flips, rotations, zooms) so the model doesn't memorize the exact training images.

**27. Why does a data augmentation layer behave differently during training vs inference?**
It's designed to apply random transformations only during training (to add variety) and pass images through unchanged during evaluation/prediction, so real-world inference isn't randomly distorted.

**28. What happens to the dense head's weights if you skip Phase 1 and go straight to Phase 2?**
The head is still randomly initialized when Phase 2 begins, so its large initial errors generate large gradients that flow into the now-unfrozen base model, risking catastrophic forgetting before the head has even stabilized.

**29. How can `model.summary()` help you verify how many parameters are trainable vs frozen?**
It prints a breakdown of "Total params," "Trainable params," and "Non-trainable params," letting you confirm the freeze/unfreeze state matches what you intended before training.

**30. What is the `initial_epoch` parameter used for when continuing from Phase 1 to Phase 2?**
It tells `fit()` what epoch number to start counting from (e.g. continuing from where Phase 1's training history left off), so epoch numbering and logs stay continuous across both phases.

## Section D — Code / Pipeline (Q31–40)

**31. Write the code to load a pretrained MobileNetV2 model without its top classification layer.**
```python
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
```

**32. Write the code to freeze the entire base model.**
```python
base_model.trainable = False
```

**33. Write the code to build a new model with `GlobalAveragePooling2D` and a `Dense` output head on top of the frozen base.**
```python
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)
```

**34. Write the code to unfreeze only the last 20 layers of a pretrained model while keeping earlier layers frozen.**
```python
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

**35. Write the `compile()` call for Phase 2 with an appropriately small learning rate.**
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**36. Write the `fit()` call for Phase 2 that continues epoch counting from where Phase 1 left off.**
```python
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    initial_epoch=history.epoch[-1]
)
```

**37. Write the code to add EarlyStopping and ModelCheckpoint callbacks and pass them into `fit()`.**
```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy', save_best_only=True
)
history = model.fit(
    train_dataset, validation_data=val_dataset,
    epochs=10, callbacks=[early_stop, checkpoint]
)
```

**38. Write a data augmentation block using `RandomFlip`, `RandomRotation`, and `RandomZoom`.**
```python
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])
```

**39. Write a small function that prints how many layers in a model are trainable vs frozen.**
```python
def count_trainable_layers(model):
    trainable = sum(1 for layer in model.layers if layer.trainable)
    frozen = sum(1 for layer in model.layers if not layer.trainable)
    print(f"Trainable layers: {trainable}, Frozen layers: {frozen}")
```

**40. List, in order, the full pipeline of steps used to go from a pretrained model to a fully fine-tuned model.**
1. Load pretrained base model with `include_top=False`
2. Freeze the entire base model
3. Attach a new head (GAP + Dense layers)
4. Compile and train (Phase 1)
5. Unfreeze the last N layers of the base model
6. Recompile with a much smaller learning rate
7. Continue training (Phase 2), optionally with `initial_epoch`
8. Use callbacks (EarlyStopping, ModelCheckpoint) throughout to monitor and save the best model

## Section E — Scenario-Based Questions (Q41–50)

**41. Scenario: You have only 300 images across 4 classes. Would you go straight to fine-tuning the whole network, or start with Phase 1? Why?**
Start with Phase 1. With so little data, unfreezing the whole network risks severe overfitting and catastrophic forgetting; freezing the base and training only a small head is much safer and needs far fewer examples.

**42. Scenario: After unfreezing the base model in Phase 2, your validation accuracy suddenly drops sharply. What's the most likely cause and how would you fix it?**
Likely the learning rate is too high for fine-tuning, causing catastrophic forgetting. Fix by lowering the learning rate significantly (e.g. to `1e-5`) and/or unfreezing fewer layers.

**43. Scenario: You forget to recompile the model after changing trainable flags. What will actually happen when you call `fit()` again?**
The optimizer will still only update the variables it was originally compiled with, so the newly unfrozen layers won't actually train even though `trainable=True` was set — the change has no effect until recompiled.

**44. Scenario: Your model overfits badly even with transfer learning and a small dataset. What are two things you could try, based on what we covered?**
Add data augmentation to increase training variety, and/or use EarlyStopping to stop training before the model starts memorizing the training set.

**45. Scenario: A teammate unfreezes the entire base model right after loading pretrained weights, skipping Phase 1 completely, and uses a learning rate of 0.001. What will likely go wrong?**
The randomly-initialized head will produce large gradients, and combined with a normal learning rate across the whole unfrozen network, this will likely destroy the pretrained features quickly — catastrophic forgetting, poor final performance.

**46. Scenario: You're transferring a model pretrained on natural images to X-ray classification. What extra concerns should you think about compared to a natural-image target task?**
X-rays differ significantly from natural images (grayscale, different textures/structures, no color cues), so higher-level features may transfer poorly — likely need to unfreeze more layers during fine-tuning, and validate carefully rather than assuming Phase 1 alone will work well.

**47. Scenario: Your training accuracy is going up but validation accuracy plateaus early in Phase 2. What callback would help you avoid wasting further epochs, and how would you configure it?**
EarlyStopping monitoring `val_loss` (or `val_accuracy`) with a reasonable `patience` (e.g. 3–5 epochs) and `restore_best_weights=True`, so training stops once no further improvement is seen and the best-performing weights are kept.

**48. Scenario: You want to save the best-performing model automatically during Phase 2 training, in case fine-tuning starts to hurt performance later. What would you set up before calling `fit()`?**
A `ModelCheckpoint` callback monitoring `val_accuracy` (or `val_loss`) with `save_best_only=True`, passed into the `callbacks` list of `fit()`.

**49. Scenario: You're asked in an interview why transfer learning works at all — why can features learned on ImageNet (cats, dogs, cars) help with a completely different task like classifying satellite images? How do you explain this?**
Early and mid-level convolutional features — edges, textures, shapes, color gradients — are largely task-agnostic and appear in almost any visual domain, not just the original classes. The model isn't just memorizing "cat vs dog," it's learning a general visual vocabulary that transfers to new tasks, especially when combined with fine-tuning the later, more task-specific layers.

**50. Scenario: Your manager asks why you use a 100x smaller learning rate in Phase 2 compared to Phase 1. Explain this in simple terms as if to a non-technical person.**
In Phase 1, we're only teaching a brand-new part of the model, so it's fine to let it learn quickly. In Phase 2, we're gently adjusting parts of a model that already knows a lot — like fine-tuning an expert's skills rather than retraining them from zero. Small, careful adjustments preserve what it already knows; big adjustments risk erasing that expertise.
