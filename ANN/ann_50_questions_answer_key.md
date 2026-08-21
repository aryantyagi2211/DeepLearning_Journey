# ANN — 50 Practice Questions: Answer Key

Use this ONLY after attempting the questions yourself first. Read the answer, understand it, close this file, and retry writing the answer in your own words before moving to the next one.

---

## Section A — Architecture & Basic Concepts

**1. What decides the number of neurons in the input layer?**
The number of features per sample (columns in your data), NOT the number of samples.

**2. What decides the number of weights in a Dense layer? Write the general formula.**
`Number of weights = (neurons in this layer) x (inputs coming into this layer)`

**3. Why does each neuron have its own independent weight(s) and bias, even within the same layer?**
So each neuron can learn to detect a different pattern from the same input. If they shared weights, every neuron would learn the exact same thing, making extra neurons pointless.

**4. What's the difference between a weight and a bias, conceptually?**
A weight controls how much influence an input has on a neuron (scales the input). A bias shifts the neuron's output up or down independently of the input — it lets the neuron activate even when input is 0, giving the model more flexibility to fit data.

**5. If a hidden layer has 5 neurons and receives 3 inputs, how many total weights and how many total biases does it have?**
Weights = 5 x 3 = 15. Biases = 5 (one per neuron).

**6. Why do we usually NOT initialize all weights to zero before training?**
If all weights start at zero, every neuron in a layer computes the exact same output and receives the exact same gradient during backprop — they'd all update identically forever, making multiple neurons useless (this is called the "symmetry problem").

**7. What is the role of an activation function? What would happen if we removed it entirely (linear network only)?**
It introduces non-linearity, allowing the network to learn complex, curved patterns instead of just straight lines. Without it, stacking multiple layers would mathematically collapse into the same as one single linear layer — no matter how many layers you add, the whole network could only ever learn straight-line relationships.

**8. Name two activation functions we used, and explain why we chose different ones for hidden layers vs the output layer.**
Sigmoid (squashes to 0-1, used in binary output layers for probability) and ReLU (used in hidden layers because it trains faster and avoids vanishing gradients in deeper networks). Sigmoid in hidden layers of deep networks can cause gradients to shrink too much during backprop.

**9. What does "fully connected" (Dense) mean?**
Every neuron in one layer connects to every neuron in the next layer.

**10. In our diagram, why did the output layer only need 1 neuron for binary classification?**
Because sigmoid outputs a single probability between 0 and 1, which is enough to represent two classes (e.g. >0.5 = class 1, less than 0.5 = class 0). Multi-class problems need more output neurons (one per class) with softmax instead.

---

## Section B — Math / Hand Calculation

**11. Write the formula for a neuron's pre-activation value `z`, labeling every symbol.**
`z = w*x + b`, where `w` = weight, `x` = input, `b` = bias, `z` = pre-activation (before applying activation function).

**12. Write the sigmoid formula and state its output range.**
`sigmoid(z) = 1 / (1 + e^(-z))`. Output range: (0, 1), never actually reaching exactly 0 or 1.

**13. Given `w=0.5, x=2, b=0.1`, calculate `z` by hand.**
`z = 0.5*2 + 0.1 = 1.0 + 0.1 = 1.1`

**14. Using your `z` from Q13, calculate `a = sigmoid(z)` by hand (approximate is fine).**
`sigmoid(1.1) = 1 / (1 + e^-1.1) ≈ 1 / (1 + 0.3329) ≈ 1/1.3329 ≈ 0.7503`

**15. Write the Binary Cross-Entropy loss formula, labeling every symbol.**
`L = -(1/n) * sum[ y_i * ln(y_hat_i) + (1-y_i) * ln(1-y_hat_i) ]`
where `n`=number of samples, `y_i`=true label, `y_hat_i`=predicted probability.

**16. If `y_true=1` and `y_pred=0.9`, is the loss high or low? Why?**
Low. The prediction (0.9) is close to the true label (1), so the model was confidently correct — BCE rewards this with a small loss value.

**17. If `y_true=0` and `y_pred=0.9`, is the loss high or low? Why?**
High. The model confidently predicted "1" (0.9 probability) when the true answer was 0 — being confidently WRONG is heavily penalized by BCE.

**18. What is the simplified gradient formula for `dL/dz` when using sigmoid + BCE together, and why does it simplify so nicely?**
`dL/dz = y_hat - y_true`. It simplifies because the derivative of BCE and the derivative of sigmoid mathematically cancel out several terms, leaving this clean expression — one of the main reasons sigmoid+BCE is such a common pairing for binary classification.

**19. Explain in your own words what backpropagation is actually doing, step by step.**
It calculates how much each weight/bias contributed to the final error (loss), by working backward from the output layer to the input layer using the chain rule — passing the "blame" for the error back through each layer, layer by layer, so every weight knows which direction to adjust to reduce the loss.

**20. Write the gradient descent weight update formula, labeling every symbol including the learning rate.**
`w_new = w_old - alpha * (dL/dw)`
where `alpha` = learning rate, `dL/dw` = gradient of loss with respect to that weight.

---

## Section C — Loss, Gradients & Training Dynamics

**21. What does it mean if loss is going DOWN but accuracy is NOT improving?**
The model's confidence/probabilities are shifting in the right general direction (reducing loss numerically), but predictions aren't yet crossing the 0.5 decision threshold to flip to the correct class. Can also signal an imbalanced dataset or a threshold/metric mismatch.

**22. What happens to training if the learning rate is set way too high?**
The model can overshoot the minimum repeatedly, causing loss to bounce around wildly or even increase/diverge instead of converging.

**23. What happens to training if the learning rate is set way too low?**
Training becomes extremely slow — tiny weight updates each step mean it could take an impractically large number of epochs to converge, or it might get stuck in a poor local minimum.

**24. What is the difference between an epoch and a batch?**
An epoch = one full pass through the ENTIRE training dataset. A batch = a smaller subset of the data used for a single forward+backward+update step. One epoch typically consists of multiple batches.

**25. If you have 1000 samples and batch_size=100, how many weight updates happen in one epoch?**
10 updates (1000 / 100 = 10 batches per epoch, one update per batch).

**26. Why do we usually shuffle data before splitting into batches?**
To avoid the model seeing data in a fixed, potentially biased order (e.g. all class-0 samples first, then all class-1) which could cause unstable or biased learning during each epoch.

**27. What's the difference between training loss and validation loss? What does it mean if training loss drops but validation loss rises?**
Training loss = error on data the model is directly learning from. Validation loss = error on data set aside to check generalization, not trained on directly. If training loss drops while validation loss rises, that's a classic overfitting sign — model is memorizing training data instead of learning generalizable patterns.

**28. Why is Adam usually preferred over plain SGD for most real-world training?**
Adam adapts the learning rate individually for each weight based on recent gradient behavior (using momentum + adaptive scaling), which usually leads to faster, more stable convergence than plain SGD's fixed learning rate for every weight.

---

## Section D — The Full Pipeline: Load to Predict

**29. Why do we need to split data into train and test sets instead of training on everything?**
To have a way to check if the model actually learned generalizable patterns, versus just memorized the training data. Testing on data the model has never seen is the only honest way to measure real-world performance.

**30. What is `random_state` actually doing, in your own words?**
It's a seed number that makes the "random" shuffle/split reproducible — same seed always produces the same shuffle order, every independent run, because the underlying pseudo-random number generator is deterministic given the same starting point.

**31. What is the difference between `random_state` and `stratify`? When would skipping `stratify` cause a real problem?**
`random_state` = makes the shuffle reproducible (same split every run). `stratify` = ensures class proportions stay balanced/consistent between train and test sets. Skipping `stratify` on an imbalanced dataset (e.g. 95% one class, 5% another) risks accidentally putting almost none of the minority class into the test set, making test accuracy misleading.

**32. What's the difference between `fit_transform()` and `transform()`? Which one do we use on test data, and why?**
`fit_transform()` learns parameters (mean/std) from the data AND applies scaling in one step. `transform()` only applies previously-learned parameters without learning anything new. We use `transform()` (alone) on test data, so test data is scaled using the SAME reference statistics as training data, never influencing those statistics itself.

**33. What would go wrong if we accidentally called `fit_transform()` on the test set instead of `transform()`?**
The test data would get scaled using its own separate mean/std, different from what the model was trained on — this is called "data leakage" and gives falsely optimistic (invalid) evaluation results, since in real deployment you won't have future data available to "fit" against ahead of time.

**34. Why is feature scaling necessary for a dataset like Breast Cancer, but wasn't needed for our toy `X=[1,2,3]` example?**
Breast Cancer has 30 features on wildly different numeric scales (some in the hundreds, some tiny decimals) — without scaling, features with bigger raw numbers would dominate weight updates unfairly. Our toy example had just 1 small, simple feature, so no scale mismatch existed.

**35. What does `validation_split=0.2` do during `model.fit()`, and how is it different from the test set?**
It carves out 20% of the TRAINING data to monitor performance after each epoch, without training on it directly — used to catch overfitting during training. It's different from the test set because it's used repeatedly during training/tuning decisions, while the test set is only checked once, at the very end, as the final unbiased measurement.

**36. What is `model.evaluate()` actually doing, mechanically (forward pass only or forward+backward)?**
Forward pass ONLY — no weight updates happen. It just computes predictions and compares them against true labels to report loss/accuracy.

**37. Why must we reshape a single new sample before calling `model.predict()` on it?**
Because Keras models always expect a batch of samples as input, even if the batch size is 1. A single flat array needs to be reshaped into a 2D shape like `(1, num_features)` to match what the model expects.

**38. Why do we use the SAME scaler (fit on training data) when scaling a brand new prediction sample, instead of fitting a new scaler?**
The model was trained on data scaled using specific mean/std values. A new sample must be scaled using those exact same reference values, otherwise the numbers would be on a different scale than what the model learned from, producing meaningless predictions.

**39. What's the purpose of `model.summary()` — what information does it give you before training even starts?**
It shows the model's architecture: each layer's type, output shape, and number of trainable parameters (weights + biases) — useful for verifying the model is built as intended before spending time training it.

**40. List, in order, all 8 pipeline steps we now use for any tabular dataset.**
Load data -> Split (train/test) -> Scale (preprocess) -> Build model -> Compile -> Train -> Evaluate -> Predict.

---

## Section E — Scenario-Based Questions

**41. Scenario: 99% training accuracy but only 65% test accuracy. What's likely happening, and 2 fixes?**
Overfitting — the model memorized training data instead of learning general patterns. Fixes: (1) reduce model complexity (fewer neurons/layers) or add regularization/dropout, (2) get more training data, or reduce number of epochs / use early stopping.

**42. Scenario: 950 benign, 50 malignant samples, model gets 95% accuracy. Should you trust this?**
No, not by itself. A model that just predicts "benign" for everything would also get ~95% accuracy without learning anything useful. Need to check other metrics like precision, recall, F1-score, or a confusion matrix to see if it's actually detecting the malignant cases correctly.

**43. Scenario: Forgot to scale features, loss barely decreases after 500 epochs. Likely cause?**
Features on very different scales are likely dominating gradient updates unevenly, causing gradient descent to struggle finding a good direction to converge — classic symptom of missing feature scaling.

**44. Scenario: Same code, same dataset, different splits every run. What's missing?**
`random_state` is not set (or set to `None`) in `train_test_split()`.

**45. Scenario: Predicting house prices (a number). Would you use sigmoid + binary_crossentropy?**
No. That combination is for binary classification. For regression (predicting a continuous number), use a linear (no) activation on the output layer, and a loss like Mean Squared Error (MSE) or Mean Absolute Error (MAE) instead.

**46. Scenario: val_loss rises after epoch 20 while loss keeps dropping to epoch 50. What should've been done around epoch 20?**
Stopped training around epoch 20 (early stopping) — continuing past that point is just overfitting further, since the model's real-world generalization (val_loss) was already getting worse.

**47. Scenario: Friend says no need to split data, just eyeball if predictions "look reasonable." What's wrong?**
Without a proper held-out test set, there's no objective, unbiased way to measure real performance — "looks reasonable" is subjective and doesn't catch overfitting, doesn't give a measurable accuracy/loss number, and doesn't scale to properly comparing different models or catching real generalization failures.

**48. Scenario: Justify using 2 hidden layers (16 then 8 neurons) instead of 1, for Breast Cancer dataset.**
With 30 real, fairly complex features, a single hidden layer might not have enough capacity to capture the relationships in the data. Two layers let the network learn broader patterns first (16 neurons) then combine/refine them into more specific representations (8 neurons) before making the final prediction — common practical architecture choice for tabular data of this size.

**49. Scenario: Used `stratify=None` on a 90/10 imbalanced dataset. What risk does this introduce?**
The random split could, by chance, place very few or even zero minority-class samples into the test set (or training set), making test accuracy unreliable/misleading and potentially leaving the model with too few minority examples to learn from during training.

**50. Scenario: Manager asks why Adam instead of plain SGD like the tiny example. Simple explanation?**
In the tiny example, we only had 7 parameters and 3 samples, so plain SGD was fast enough. Real datasets have far more parameters and samples, so Adam's ability to automatically adjust the learning rate per-weight based on recent gradient behavior makes training noticeably faster and more stable — a practical necessity at real-world scale, not just a small dataset toy problem.

---

*Once you've retried all 50 in your own words, we can go over any you're still unsure about together.*
