# ANN — 50 Practice Questions

Covers everything from today + previous ANN sessions: architecture, math/backprop, activation functions, loss, the full training pipeline (load → split → scale → build → compile → train → evaluate → predict), and scenario-based reasoning.

Try answering these yourself first — no answer key given on purpose, so you actually think them through. We can go over your answers together after.

---

## Section A — Architecture & Basic Concepts (Q1–10)

1. What decides the number of neurons in the input layer?
2. What decides the number of weights in a Dense layer? Write the general formula.
3. Why does each neuron have its own independent weight(s) and bias, even within the same layer?
4. What's the difference between a weight and a bias, conceptually?
5. If a hidden layer has 5 neurons and receives 3 inputs, how many total weights and how many total biases does it have?
6. Why do we usually NOT initialize all weights to zero before training?
7. What is the role of an activation function? What would happen if we removed it entirely (linear network only)?
8. Name two activation functions we used, and explain why we chose different ones for hidden layers vs the output layer.
9. What does "fully connected" (Dense) mean?
10. In our diagram, why did the output layer only need 1 neuron for binary classification?

---

## Section B — Math / Hand Calculation (Q11–20)

11. Write the formula for a neuron's pre-activation value `z`, labeling every symbol.
12. Write the sigmoid formula and state its output range.
13. Given `w=0.5, x=2, b=0.1`, calculate `z` by hand.
14. Using your `z` from Q13, calculate `a = sigmoid(z)` by hand (approximate is fine).
15. Write the Binary Cross-Entropy loss formula, labeling every symbol.
16. If `y_true=1` and `y_pred=0.9`, is the loss high or low? Why?
17. If `y_true=0` and `y_pred=0.9`, is the loss high or low? Why?
18. What is the simplified gradient formula for `dL/dz` when using sigmoid + BCE together, and why does it simplify so nicely?
19. Explain in your own words what backpropagation is actually doing, step by step.
20. Write the gradient descent weight update formula, labeling every symbol including the learning rate.

---

## Section C — Loss, Gradients & Training Dynamics (Q21–28)

21. What does it mean if loss is going DOWN but accuracy is NOT improving?
22. What happens to training if the learning rate is set way too high?
23. What happens to training if the learning rate is set way too low?
24. What is the difference between an epoch and a batch?
25. If you have 1000 samples and batch_size=100, how many weight updates happen in one epoch?
26. Why do we usually shuffle data before splitting into batches?
27. What's the difference between training loss and validation loss? What does it mean if training loss drops but validation loss rises?
28. Why is Adam usually preferred over plain SGD for most real-world training?

---

## Section D — The Full Pipeline: Load → Predict (Q29–40)

29. Why do we need to split data into train and test sets instead of training on everything?
30. What is `random_state` actually doing, in your own words?
31. What is the difference between `random_state` and `stratify`? When would skipping `stratify` cause a real problem?
32. What's the difference between `fit_transform()` and `transform()`? Which one do we use on test data, and why?
33. What would go wrong if we accidentally called `fit_transform()` on the test set instead of `transform()`?
34. Why is feature scaling necessary for a dataset like Breast Cancer, but wasn't needed for our toy `X=[1,2,3]` example?
35. What does `validation_split=0.2` do during `model.fit()`, and how is it different from the test set?
36. What is `model.evaluate()` actually doing, mechanically (forward pass only or forward+backward)?
37. Why must we reshape a single new sample before calling `model.predict()` on it?
38. Why do we use the SAME scaler (fit on training data) when scaling a brand new prediction sample, instead of fitting a new scaler?
39. What's the purpose of `model.summary()` — what information does it give you before training even starts?
40. List, in order, all 8 pipeline steps we now use for any tabular dataset.

---

## Section E — Scenario-Based Questions (Q41–50)

41. **Scenario:** You train a model and get 99% training accuracy but only 65% test accuracy. What's likely happening, and name 2 things you could try to fix it.

42. **Scenario:** Your dataset has 950 samples labeled "benign" and only 50 labeled "malignant." Your model gets 95% accuracy. Should you trust this number? Why or why not?

43. **Scenario:** You forgot to scale your features before training. Training runs, but loss barely decreases even after 500 epochs. What's the most likely cause?

44. **Scenario:** Two teammates run the exact same code on the exact same dataset, but get different train/test splits every time they run it. What's missing from their code?

45. **Scenario:** You're building a model to predict house prices (a number, not a class). Would you use `sigmoid` + `binary_crossentropy` for the output layer? If not, what would you use instead?

46. **Scenario:** During training, you notice `val_loss` starts increasing after epoch 20, while `loss` keeps decreasing all the way to epoch 50. What should you have done around epoch 20?

47. **Scenario:** A friend says "I don't need to split my data into train/test, I'll just check if my model's predictions look reasonable." Explain what's wrong with this approach.

48. **Scenario:** You're asked to justify why you used 2 hidden layers (16 neurons, then 8 neurons) instead of just 1 hidden layer for the Breast Cancer dataset. What would you say?

49. **Scenario:** You accidentally use `stratify=None` on a dataset where 90% of samples are one class. What specific risk does this introduce into your train/test split?

50. **Scenario:** Your manager asks: "Why did we use Adam optimizer instead of plain SGD like we used in the tiny example?" How do you explain this in simple terms?

---

*Try answering as many as you can. We'll go through your answers together and correct/clarify anything needed before moving to CNN.*
