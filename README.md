# 🏺 Arabic Poetry Generation using Transformer Encoder

This project explores the development of an **Arabic poetry generation model** using a **Transformer Encoder** architecture adapted for classification.  
The goal is to predict the next **character** or **word**, effectively generating text sequence-by-sequence.  

Two modeling approaches were implemented:

- **Character-Level Generation**
- **Word-Level Generation**

---

## 📘 Project Summary

| Feature | Character-Level Model | Word-Level Model |
|----------|----------------------|------------------|
| **Input/Output Unit** | Single Arabic Character | Single Arabic Word (Token) |
| **Token Count** | Very small (~35 tokens) | Large (~73,000 tokens) |
| **Preprocessing** | Basic Arabic normalization, diacritic removal | Advanced token decomposition (prefixes, suffixes, merges) and normalization |
| **Model Architecture** | Encoder v2 Compact Fixed | Encoder v1 (Two Encoder Blocks with Weight Tying) |
| **Window Size** | 100 Characters | 64 Words (+1 for CLS) |
| **Result** | Produces syntactically correct-looking text but limited semantic coherence | Much better poetic structure and coherence |

---

## 🧹 1. Data and Preprocessing

The foundation for both models was the **Arabic Poetry Dataset** from [Kaggle](https://www.kaggle.com/).

### A. Data Loading & Cleaning

- **Source:** `Arabic_poetry_dataset.csv`  
- **Normalization:** Applied a rigorous `clean_poem_text()` function across all poem texts, including:
  - Normalization of Arabic letters (e.g., `أ`, `ئ`, `ى` → standard forms)
  - Removal of **Tatweel (ـ)**
  - Removal of **diacritics (Tashkeel)** — `[ \u064B-\u0652 ]`
  - Removal of **English letters** and **digits**
  - Standardization of **spacing** and **newlines**

### B. Tokenization & Sequence Preparation

#### 🔹 Character-Level Approach (Lower Quality)
- **Vocabulary:** Dictionary of all unique characters (≈35), including punctuation and `START_TOKEN`.  
- **Sequence Generation:**  
  - Window Size = 100 characters  
  - Each 100-character sequence predicts the **101st character**.

#### 🔹 Word-Level Approach (Higher Quality – Preferred)
- **Advanced Decomposition (Crucial Step):**
  - Split words into sub-word units by separating known prefixes (`وال`, `فال`, `ال`, `و`, `ف`, `ب`, etc.)  
    and suffixes (`هم`, `هن`, `ها`, `ك`, `ي`, etc.)  
  - Used a dedicated `CONCAT_TOKEN` to maintain connections.
- **Tokenizer:**  
  - Built with **Keras Tokenizer** including `OOV_TOKEN`, `START_TOKEN`, `CONCAT_TOKEN`, `CLS_TOKEN`.
- **Sequence Generation:**  
  - 64-word input window + 1 `CLS_TOKEN`
  - Sequences padded with `START_TOKEN` as needed.

---

## 🧠 2. Model Architecture and Training

Both models utilize **custom Transformer Encoder architectures**.

### A. Character-Level Model — *Encoder v2 Compact Fixed*

**Objective:** Next Character Prediction (Classification)

**Architecture:**
1. Input layer with appended `CLS_TOKEN`
2. Embedding (32-dim)
3. Parallel **Conv1D layers** (kernels 1, 3, 5, 7) → concatenated for local context
4. **Positional Encoding**
5. **Multi-Head Attention (8 heads)**
6. **Layer Normalization + Skip Connections**
7. Extracted `CLS_TOKEN` → Cross-product layer → Dense Feed-Forward
8. **Softmax output** over ~35-character vocabulary

**Training Metrics:**
- Validation Accuracy: **~48.6%** after 20 epochs

---

### B. Word-Level Model — *Encoder v1 (Recommended)*

**Objective:** Next Word/Sub-word Prediction (Classification)

**Architecture:**
1. Input: 65 tokens (1 CLS + 64 words)
2. Embedding (144-dim)
3. Positional Encoding + Dropout
4. **Two Transformer Encoder Blocks**
   - Multi-Head Attention
   - Dropout + Skip Connections
   - Layer Normalization + Feed-Forward
5. Extracted `CLS_TOKEN`
6. **Weight Tying** — Final projection multiplies hidden state by embedding matrix transpose
7. **Softmax output** over ~73,000-token vocabulary

**Training Metrics:**
- Validation Accuracy: **~55%** after 10 epochs

---

## 🏁 3. Conclusion: Level Choice

| Aspect | Character-Level | Word-Level |
|--------|------------------|------------|
| **Speed of Learning** | Quickly learns character legality | Slower but learns structure |
| **Semantic Coherence** | Poor | Strong |
| **Output Quality** | Often gibberish after a few words | Coherent and poetic |
| **Recommended** | ❌ | ✅ |

➡️ The **Word-Level model** clearly outperforms the Character-Level one.  
Thanks to advanced decomposition and larger vocabulary, it captures **semantic relationships** and **poetic structure** more effectively.

---

## 📊 4. Visualization & Outputs

| Description | Filepath |
|--------------|-----------|
| Word-Level Model Architecture | *(add path here)* |
| Character-Level Model Architecture | *(add path here)* |
| Word-Level Training History (Loss/Accuracy) | *(add path here)* |
| Character-Level Training History (Loss/Accuracy) | *(add path here)* |
| Poem Line Distribution Plot | *(add path here)* |

---

## 🧾 Author
**Developed by:** Youssef Samy  
**Field:** Machine Learning & NLP  
**Project:** Arabic Poetry Generation using Transformer Encoder  
**Language:** Python (TensorFlow / Keras)

---

## 🧩 Future Work
- Integrate **Transformer Decoder** for full seq-to-seq generation.  
- Experiment with **GPT-style autoregressive** architectures.  
- Apply **fine-tuning** on modern and classical Arabic corpora for style control.

---
