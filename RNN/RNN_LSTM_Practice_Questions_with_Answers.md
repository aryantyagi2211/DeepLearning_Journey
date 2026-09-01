# RNN & LSTM — 50 Practice Questions (With Answers)

---

## Section A — RNN Concept (Q1–Q12)

**1. What is the main difference between how an ANN processes input and how an RNN processes input?**
ANN treats every input as independent, with no memory of previous inputs. RNN processes input one step at a time (sequentially) and keeps a memory (hidden state) of everything seen so far.

**2. What is the "hidden state" in an RNN, in your own words?**
A short mental summary that gets updated after every timestep — it represents "everything important I've understood so far" in a fixed-size vector.

**3. Why is the hidden state always a fixed size, no matter how long the input sequence is?**
Because the same weight matrices and the same formula are applied at every timestep — the architecture doesn't grow with sequence length, it just keeps updating the same-sized vector.

**4. What happens to the hidden state every time a new word/timestep arrives?**
It gets combined with the new input and overwritten — the old hidden state is replaced by a new one that blends old + new information.

**5. In `h(t) = tanh(Wxh·x(t) + Whh·h(t-1) + b)`, what does each symbol stand for?**
h(t) = new hidden state, Wxh = weight matrix for input, x(t) = current input, Whh = weight matrix for previous hidden state, h(t-1) = previous hidden state, b = bias, tanh = squashing activation function.

**6. Why are the same weights (Wxh, Whh) reused at every timestep instead of learning new weights for each position?**
Because the network should apply the same "combining logic" regardless of position in the sequence — this is called weight sharing, and it also lets RNN handle sequences of any length with a fixed number of parameters.

**7. If a sentence has 200 words, does the RNN store all 200 words separately in memory? Explain.**
No. It compresses all 200 words into ONE fixed-size hidden state vector. Individual words are not stored separately — their influence is blended together, with older words fading more.

**8. What does it mean when we say RNN "blends" information instead of "storing" it?**
Each new hidden state is a mix of the old hidden state and the new input — like mixing paint colors. You cannot cleanly extract "what word 5 was" from the final hidden state; only its diluted influence remains.

**9. Why are recent words remembered better than older words in a vanilla RNN?**
Because older words have been through more rounds of blending/overwriting, so their original influence has been multiplied down repeatedly and has shrunk more.

**10. What is "fading" in the context of RNN memory?**
The mathematical shrinking of an old word's contribution/strength toward zero as more new words get mixed into the hidden state over time.

**11. Is RNN's fading based on how important a word is? Why or why not?**
No — it is purely based on distance/position. RNN has no built-in concept of "importance"; it fades everything at roughly the same rate regardless of meaning.

**12. Name two real-world domains where RNN/LSTM models are still used today, and explain why.**
Example answers: (1) Time-series forecasting (stock prices, demand forecasting) — data is naturally sequential and models are lightweight. (2) Predictive maintenance / sensor data — real-time streaming data where you can't wait for a full sequence like Transformers need. (Other valid answers: fraud detection, wearables, some speech tasks.)

---

## Section B — Vanishing Gradient Problem (Q13–Q18)

**13. What is the "vanishing gradient problem"?**
During training, the gradient (error signal used to update weights) shrinks exponentially as it's propagated backward through many timesteps, until it becomes close to zero for early timesteps — making it very hard for the network to learn long-term dependencies.

**14. Why does the vanishing gradient problem get worse for longer sequences?**
Because the gradient gets multiplied by a small factor at every timestep it passes through. The more timesteps, the more multiplications, and the closer the result gets to zero (exponential decay).

**15. If a hidden state value gets multiplied by 0.5 at every timestep, what is its strength after 6 timesteps (starting from 1.0)?**
1.0 × 0.5⁶ = 1.0 × 0.015625 ≈ **0.0156**

**16. Why can't vanilla RNN reliably answer a question about something mentioned 100 words earlier?**
Because that information's influence in the hidden state has faded almost to zero by the time 100 more words have been processed — the signal is too weak to recover.

**17. What role does the tanh activation play in causing the vanishing gradient problem?**
tanh squashes values into the range (-1, 1) at every single timestep. Repeatedly squashing the hidden state (and its gradient during backprop) causes values to shrink toward zero over many steps.

**18. What was the vanishing gradient problem the main motivation for solving, historically?**
It was the main motivation for inventing LSTM (and later GRU) — architectures designed specifically to let gradients and information flow across long sequences without vanishing.

---

## Section C — LSTM Concept (Q19–Q30)

**19. What does LSTM stand for?**
Long Short-Term Memory.

**20. What are the TWO types of memory in an LSTM, and what does each one represent?**
Cell state (C) — long-term protected memory ("the notebook"). Hidden state (h) — short-term working memory/output, similar in role to RNN's hidden state.

**21. Why is the cell state described as "protected" compared to RNN's hidden state?**
Because it is only modified through explicit, learned gate operations (multiply by forget gate, add scaled candidate values) — it isn't squashed through an activation function at every single step like RNN's hidden state is.

**22. Name the three gates in an LSTM and, in one line each, what decision each gate makes.**
Forget gate — decides how much of the old cell state to keep vs erase. Input gate — decides how much new candidate information to add to the cell state. Output gate — decides how much of the cell state to reveal as the current hidden state.

**23. What does it mean when a gate outputs a value close to 0? What about close to 1?**
Close to 0 means "block/erase almost everything" (let almost nothing through). Close to 1 means "let almost everything through unchanged."

**24. Why do LSTM gates use sigmoid activation instead of tanh?**
Because sigmoid outputs values between 0 and 1, which naturally represent a percentage/proportion — perfect for a gate that decides "how much" to let through. tanh's range (-1 to 1) doesn't map to that "how much to allow" interpretation.

**25. Why does the candidate value (C~) use tanh instead of sigmoid?**
Because the candidate represents actual new information/content to be added (not a gating decision), and tanh's range (-1 to 1) allows it to add or subtract in a balanced way, similar to how RNN's hidden state activation works.

**26. In the "Ariel" example, which gate is responsible for deciding to write "Ariel" into the notebook?**
The input gate (together with the candidate values proposing "Ariel" as worth storing).

**27. In the same example, which gate is responsible for keeping "Ariel" from being erased later?**
The forget gate — by outputting a value close to 1 for that part of the cell state, it lets "Ariel" pass through mostly unchanged at each subsequent timestep.

**28. What is the key structural reason the cell state fades much slower than RNN's hidden state?**
The cell state update is just multiply (forget gate) and add (input gate × candidate) — no activation function squashes the entire cell state at every step, unlike RNN where the whole hidden state passes through tanh every single timestep.

**29. Are LSTM gates fixed rules written by a programmer, or are they learned? Explain.**
They are learned. The weights (W, U, b) inside each gate's formula are trained via backpropagation, so the network learns on its own which patterns are worth protecting or forgetting.

**30. Give one real-world sentence example (different from "Ariel") where an LSTM's forget gate would need to protect an important word for a long time.**
Example answer: "The password is Blue42Sky. [...long unrelated conversation...] What was the password again?" — the forget gate would need to protect "Blue42Sky" across many irrelevant timesteps.

---

## Section D — Formula & Terminology (Q31–Q38)

**31. In `gate = sigmoid(W·h(t-1) + U·x(t) + b)`, what is W responsible for weighting?**
The previous hidden state h(t-1) — how much the past short-term memory matters to this gate's decision.

**32. In the same formula, what is U responsible for weighting?**
The current input x(t) — how much the current word/timestep matters to this gate's decision.

**33. What is the "pre-activation" (commonly called z) in this formula?**
The raw combined score before squashing: z = W·h(t-1) + U·x(t) + b — this is what gets passed into sigmoid (or tanh for the candidate).

**34. Why is the raw score passed through sigmoid instead of being used directly as the gate value?**
Because the raw score (z) can be any real number (positive, negative, large, small), but a gate needs to represent "how much to let through" as a clean value between 0 and 1 — sigmoid provides exactly that squashing.

**35. What is the difference between W and U in terms of naming conventions used in different papers/textbooks?**
Some sources call them W (for hidden state) and U (for input); others use Whf/Wxf-style subscript notation (e.g., Whf = weight on hidden state for forget gate, Wxf = weight on input for forget gate). Same underlying concept, different labeling styles.

**36. Write the formula for the forget gate f(t) using proper notation.**
f(t) = sigmoid(Wf·h(t-1) + Uf·x(t) + bf)

**37. Write the formula for the candidate values C~(t) using proper notation.**
C~(t) = tanh(Wc·h(t-1) + Uc·x(t) + bc)

**38. Write the formula for how the new cell state C(t) is calculated from the forget gate, input gate, and candidate values.**
C(t) = f(t)·C(t-1) + i(t)·C~(t)

---

## Section E — Hand Calculation (Q39–Q46)

*Given: h(t-1) = 0.5, C(t-1) = 0.8, x(t) = 1.0*

**39. Using weights Wf=0.5, Uf=0.3, bf=0.1, calculate z_f.**
z_f = (0.5 × 0.5) + (0.3 × 1.0) + 0.1 = 0.25 + 0.3 + 0.1 = **0.65**

**40. Using z_f from Q39, calculate f(t) = sigmoid(z_f).**
f(t) = sigmoid(0.65) ≈ **0.657**

**41. Using weights Wi=0.4, Ui=0.6, bi=-0.1, calculate i(t).**
z_i = (0.4 × 0.5) + (0.6 × 1.0) − 0.1 = 0.2 + 0.6 − 0.1 = 0.7
i(t) = sigmoid(0.7) ≈ **0.668**

**42. Using weights Wc=0.3, Uc=0.5, bc=0.05, calculate C~(t).**
z_c = (0.3 × 0.5) + (0.5 × 1.0) + 0.05 = 0.15 + 0.5 + 0.05 = 0.7
C~(t) = tanh(0.7) ≈ **0.604**

**43. Using your answers from Q40–Q42, calculate C(t) = f(t)·C(t-1) + i(t)·C~(t).**
C(t) = (0.657 × 0.8) + (0.668 × 0.604) = 0.526 + 0.404 ≈ **0.929**

**44. Using weights Wo=0.6, Uo=0.2, bo=0.0, calculate o(t).**
z_o = (0.6 × 0.5) + (0.2 × 1.0) + 0 = 0.3 + 0.2 = 0.5
o(t) = sigmoid(0.5) ≈ **0.622**

**45. Using your answers from Q43 and Q44, calculate h(t) = o(t)·tanh(C(t)).**
tanh(0.929) ≈ 0.730
h(t) = 0.622 × 0.730 ≈ **0.455**

**46. If the forget gate output was 0.1 instead of 0.657 (everything else the same), would the old cell state be mostly kept or mostly erased? What does this tell you about what the network "decided"?**
Mostly erased — only 10% of C(t-1) would carry forward (C(t) = 0.1×0.8 + 0.668×0.604 = 0.08 + 0.404 ≈ 0.484, noticeably lower than 0.929). This tells us the network "decided" that whatever is stored in the old cell state is no longer relevant and should be mostly discarded in favor of new information.

---

## Section F — RNN vs LSTM Comparison (Q47–Q50)

**47. Fill in the blank: RNN has ONE memory that gets ______ at every step, while LSTM has TWO memories — one protected and one short-term.**
**overwritten** (completely replaced/blended at every timestep)

**48. Why is LSTM able to remember important information across 100+ words while vanilla RNN struggles?**
Because LSTM's cell state is only modified through gated multiply/add operations (not squashed every step), and the forget gate can learn to output values close to 1 for important information — letting it pass through largely unchanged across many timesteps, unlike RNN's hidden state which fades everything equally.

**49. What is the single biggest structural addition LSTM makes on top of vanilla RNN's architecture?**
The cell state (long-term memory highway) controlled by three learned gates (forget, input, output) — giving the network an explicit, trainable mechanism to decide what to remember and what to forget.

**50. In one or two sentences, explain why LSTM's full name — "Long Short-Term Memory" — makes sense given what it actually does.**
It maintains a short-term memory (hidden state, h) just like RNN, but adds a mechanism for long-term memory (the gated cell state, C) that can preserve important information across many timesteps — combining both "long-term" and "short-term" memory in one architecture, hence the name.
