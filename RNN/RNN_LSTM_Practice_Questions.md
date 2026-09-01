# RNN & LSTM — 50 Practice Questions

*Mix of concept, terminology, and hand-calculation questions. No answers in this file — see `RNN_LSTM_Practice_Questions_with_Answers.md` for the answer key.*

---

## Section A — RNN Concept (Q1–Q12)

1. What is the main difference between how an ANN processes input and how an RNN processes input?
2. What is the "hidden state" in an RNN, in your own words?
3. Why is the hidden state always a fixed size, no matter how long the input sequence is?
4. What happens to the hidden state every time a new word/timestep arrives?
5. In the formula `h(t) = tanh(Wxh·x(t) + Whh·h(t-1) + b)`, what does each symbol stand for?
6. Why are the same weights (Wxh, Whh) reused at every timestep instead of learning new weights for each position?
7. If a sentence has 200 words, does the RNN store all 200 words separately in memory? Explain.
8. What does it mean when we say RNN "blends" information instead of "storing" it?
9. Why are recent words remembered better than older words in a vanilla RNN?
10. What is "fading" in the context of RNN memory?
11. Is RNN's fading based on how important a word is? Why or why not?
12. Name two real-world domains where RNN/LSTM models are still used today, and explain why.

## Section B — Vanishing Gradient Problem (Q13–Q18)

13. What is the "vanishing gradient problem"?
14. Why does the vanishing gradient problem get worse for longer sequences?
15. If a hidden state value gets multiplied by 0.5 at every timestep, what is its strength after 6 timesteps (starting from 1.0)?
16. Why can't vanilla RNN reliably answer a question about something mentioned 100 words earlier?
17. What role does the tanh activation play in causing the vanishing gradient problem?
18. What was the vanishing gradient problem the main motivation for solving, historically?

## Section C — LSTM Concept (Q19–Q30)

19. What does LSTM stand for?
20. What are the TWO types of memory in an LSTM, and what does each one represent?
21. Why is the cell state described as "protected" compared to RNN's hidden state?
22. Name the three gates in an LSTM and, in one line each, what decision each gate makes.
23. What does it mean when a gate outputs a value close to 0? What about close to 1?
24. Why do LSTM gates use sigmoid activation instead of tanh?
25. Why does the candidate value (C~) use tanh instead of sigmoid?
26. In the "Ariel" example, which gate is responsible for deciding to write "Ariel" into the notebook?
27. In the same example, which gate is responsible for keeping "Ariel" from being erased later?
28. What is the key structural reason the cell state fades much slower than RNN's hidden state?
29. Are LSTM gates fixed rules written by a programmer, or are they learned? Explain.
30. Give one real-world sentence example (different from "Ariel") where an LSTM's forget gate would need to protect an important word for a long time.

## Section D — Formula & Terminology (Q31–Q38)

31. In `gate = sigmoid(W·h(t-1) + U·x(t) + b)`, what is W responsible for weighting?
32. In the same formula, what is U responsible for weighting?
33. What is the "pre-activation" (commonly called z) in this formula?
34. Why is the raw score passed through sigmoid instead of being used directly as the gate value?
35. What is the difference between W and U in terms of naming conventions used in different papers/textbooks?
36. Write the formula for the forget gate f(t) using proper notation.
37. Write the formula for the candidate values C~(t) using proper notation.
38. Write the formula for how the new cell state C(t) is calculated from the forget gate, input gate, and candidate values.

## Section E — Hand Calculation (Q39–Q46)

Use these values for all questions in this section unless stated otherwise:
`h(t-1) = 0.5, C(t-1) = 0.8, x(t) = 1.0`

39. Using weights Wf=0.5, Uf=0.3, bf=0.1, calculate the forget gate's pre-activation value z_f.
40. Using the z_f from Q39, calculate f(t) = sigmoid(z_f). (sigmoid(0.65) ≈ 0.657)
41. Using weights Wi=0.4, Ui=0.6, bi=-0.1, calculate the input gate value i(t).
42. Using weights Wc=0.3, Uc=0.5, bc=0.05, calculate the candidate value C~(t).
43. Using your answers from Q40–Q42, calculate the new cell state C(t) = f(t)·C(t-1) + i(t)·C~(t).
44. Using weights Wo=0.6, Uo=0.2, bo=0.0, calculate the output gate value o(t).
45. Using your answers from Q43 and Q44, calculate the final hidden state h(t) = o(t)·tanh(C(t)).
46. If the forget gate output was 0.1 instead of 0.657 (everything else the same), would the old cell state be mostly kept or mostly erased? What does this tell you about what the network "decided"?

## Section F — RNN vs LSTM Comparison (Q47–Q50)

47. Fill in the blank: RNN has ONE memory that gets ______ at every step, while LSTM has TWO memories — one protected and one short-term.
48. Why is LSTM able to remember important information across 100+ words while vanilla RNN struggles?
49. What is the single biggest structural addition LSTM makes on top of vanilla RNN's architecture?
50. In one or two sentences, explain why LSTM's full name — "Long Short-Term Memory" — makes sense given what it actually does.
