# NLP Journey — Topic 1: Text Preprocessing (Q&A)

20 questions covering all 4 subtopics: Tokenization, Stemming vs Lemmatization, Stopwords Removal, Text Cleaning/Normalization.

---

**Q1. What is the main purpose of tokenization?**
Answer: To break raw text into smaller meaningful units.
Explanation: Tokenization splits raw text into discrete units (tokens) so downstream steps can process them.

---

**Q2. Which tokenization level splits text like ["I", "'m", "learning", "NLP", "."]?**
Answer: Word-level.
Explanation: Word-level tokenization splits text into individual words and punctuation marks.

---

**Q3. Why is subword tokenization (BPE, WordPiece) preferred in modern Transformers/LLMs?**
Answer: It handles rare/unseen words and keeps vocabulary size manageable.
Explanation: Subword tokenization breaks rare or long words into known smaller pieces, avoiding huge vocabularies and handling unseen words gracefully.

---

**Q4. In the example "learning" -> ["learn", "##ing"], what does the "##" typically indicate?**
Answer: This piece continues from the previous token, not a new word.
Explanation: The ## prefix (used in WordPiece) marks a subword continuation of the previous token rather than a standalone word start.

---

**Q5. Which of the following is a TRUE statement about stemming?**
Answer: It uses rule-based suffix chopping and can produce non-words.
Explanation: Stemming applies fixed rules to strip suffixes without checking a dictionary, so results like "studi" may not be real words.

---

**Q6. What does lemmatization use that stemming does not?**
Answer: A dictionary and part-of-speech context.
Explanation: Lemmatization looks up words in a dictionary and considers their POS to return a valid base form (lemma).

---

**Q7. Why does lemmatization convert "better" to "good"?**
Answer: It recognizes "better" as the comparative form of "good".
Explanation: Lemmatization understands grammatical relationships, so it maps comparative/superlative forms back to their base lemma.

---

**Q8. When would you prefer stemming over lemmatization?**
Answer: When speed matters more than precision, e.g. large-scale search indexing.
Explanation: Stemming is faster and rule-based, making it suitable for large-scale, speed-sensitive tasks like search indexing.

---

**Q9. What are stopwords?**
Answer: High-frequency words that carry little topic-specific meaning, like "is", "a", "the".
Explanation: Stopwords are common, low-information words that typically get removed to reduce noise in text.

---

**Q10. In the sentence ["this", "is", "a", "great", "movie"], which tokens would typically be removed as stopwords?**
Answer: "this", "is", "a".
Explanation: These are common stopwords with little topic signal; "great" and "movie" carry the actual meaning and are kept.

---

**Q11. Why is stopword removal useful before Bag of Words / TF-IDF representation?**
Answer: It prevents high-frequency, low-information words from dominating word counts.
Explanation: Without removing stopwords, frequent but meaningless words would dominate frequency-based representations, diluting useful signal.

---

**Q12. What does lowercasing prevent in text processing?**
Answer: "Movie" and "movie" being treated as two different tokens.
Explanation: Lowercasing standardizes case so the same word isn't split into multiple distinct tokens due to capitalization differences.

---

**Q13. Which of these is an example of the text cleaning/normalization step?**
Answer: Removing punctuation and handling emojis/hashtags.
Explanation: Text cleaning/normalization covers lowercasing, punctuation removal, and handling special characters like emojis and hashtags.

---

**Q14. What is the correct order of the 4 subtopics in Text Preprocessing as taught?**
Answer: Tokenization -> Stemming/Lemmatization -> Stopwords removal -> Cleaning/Normalization.
Explanation: The pipeline order taught was: Tokenization, then Stemming vs Lemmatization, then Stopwords removal, then Text cleaning/normalization.

---

**Q15. Why can't a model work directly on raw text without any preprocessing?**
Answer: Raw text is just a sequence of characters with no inherent discrete structure for the model to use.
Explanation: Text needs to be broken into structured, discrete units before any counting, representation, or modeling can happen.

---

**Q16. Which subword tokenization algorithms were mentioned as examples?**
Answer: BPE, WordPiece, SentencePiece.
Explanation: These are common subword tokenization algorithms used in modern NLP/LLMs.

---

**Q17. What problem does reducing words to a common root (via stemming/lemmatization) solve?**
Answer: It prevents the model from wasting capacity learning that "study" and "studying" mean the same thing separately.
Explanation: Grouping related word forms under one root reduces redundant vocabulary and helps the model see them as the same underlying concept.

---

**Q18. In the stopwords removal diagram, what determines whether a token is discarded or kept?**
Answer: Whether it matches an entry in the stopword list.
Explanation: Each token is checked against a predefined stopword list; a match means it gets discarded, otherwise it's kept.

---

**Q19. Which stage comes right before a cleaned/normalized text re-enters the pipeline for further steps like representation (BoW/TF-IDF)?**
Answer: Text cleaning/normalization is the last preprocessing step shown.
Explanation: Text cleaning/normalization was the 4th and final subtopic of Text Preprocessing, producing clean text ready for the next topic (Text Representation).

---

**Q20. Overall, what is the main goal of the entire Text Preprocessing stage in the NLP pipeline?**
Answer: To convert messy raw text into clean, structured, discrete tokens ready for representation and modeling.
Explanation: Text preprocessing's overall job is to transform noisy raw text into a clean, tokenized form that later stages (representation, embeddings, modeling) can use effectively.
