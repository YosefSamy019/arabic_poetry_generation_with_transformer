# 🏺 Arabic Poetry Generation using Transformer Encoder

This project explores the development of an **Arabic poetry generation model** using a **Transformer Encoder** architecture adapted for text classification and next-token prediction.
The goal is to predict the next **character** or **word**, effectively generating poetic text sequence-by-sequence.

Two modeling paths were explored:

* **Character-Level Model**
* **Word-Level Model (Implemented in this version)** ✅

---

## 🚀 Deployment

**Streamlit Demo:**
👉 [Arabic Poetry Generation App](https://arabicpoetrygenerationwithtransformer-mherlxkzk8kxwecj5yanv2.streamlit.app/)

---

## 📘 Project Summary

| Feature                  | Character-Level Model           | Word-Level Model *(Implemented)*            |
| ------------------------ | ------------------------------- | ------------------------------------------- |
| **Input/Output Unit**    | Single Arabic Character         | Single Arabic Word                          |
| **Tokenizer Vocabulary** | ~35 tokens                      | ~73,000 tokens                              |
| **Architecture**         | Encoder v2 Compact Fixed        | Encoder v1 (2 Encoder Blocks, Weight Tying) |
| **Sequence Window**      | 100 Characters + CLS            | 64 Words + CLS                              |
| **Result**               | Valid syntax but poor semantics | Fluent and coherent poetic lines            |

---

## 🧹 1. Data Preparation and Preprocessing

### 📜 Dataset

The model was trained on the **Arabic Poetry Dataset** from [Kaggle](https://www.kaggle.com/datasets/ahmedabelal/arabic-poetry).

### 🧼 Cleaning and Normalization

The function `clean_poem_text()` performs deep normalization:

* Converts `أ`, `إ`, `آ` → `ا`, `ة` → `ه`
* Removes **Tatweel (ـ)** and **Tashkeel (diacritics)** `[\u064B-\u0652]`
* Removes digits, English letters, and foreign symbols
* Ensures proper spacing around punctuation and newlines
* Decomposes **prefixes** (`ال`, `وال`, `ف`, `ك`, `ب`, `ل`)
  and **suffixes** (`هم`, `ها`, `ي`, `نا`)
  using a concatenation token `&`
* Resolves merged words like `"هواءتالف"` → `"هواء & تالف"`
* Adds `\n` handling for poem structure separation

### 🧩 Tokenization

A **Keras Tokenizer** is used with:

* `START_TOKEN` (`#`)
* `CONCAT_TOKEN` (`&`)
* `CLS_TOKEN` (`$`)
* `OOV_TOKEN` for out-of-vocabulary words

**Outputs saved in `/utils/`:**

* `tokenizer.pickle` — serialized tokenizer
* `tokens.json` — vocabulary index mapping

**Windowing:**

* Input sequence = 64 tokens
* Label = next token (word)
* CLS token appended to each window

---

## 🧾 2. Data Scanning and Loader

### 🔍 Poem Scanning

All poems are first converted into supervised training samples using a **sliding window approach**.
Each poem generates overlapping sequences of words, forming `(input_window → next_token)` pairs.

Example:

```
poem = "يا ليل الصب متى غده"
windows:
  ["يا", "ليل", "الصب", "متى"] → "غده"
  ["ليل", "الصب", "متى", "غده"] → "..."
```

This creates a structured **`scanned_df`** dataset.

### ⚙️ DataLoader

A custom `DataLoader` built from `tf.keras.utils.Sequence` manages all training data dynamically.

Key Features:

* **Inputs:**

  * `scanned_df`, `df`, `tokenizer_dict`
  * `batch_size`, `shuffle`, `drop_random`, `start_token`
* **Global Cache:** `_GLOBAL_POEM_CACHE` to avoid reloading poems repeatedly.
* **Dynamic Windowing:** Generates padded, tokenized samples per batch.
* **Drop Random:** Randomly skips a portion of data for regularization.
* **Shuffling:** Ensures different poem ordering every epoch.

This ensures memory-efficient training even for tens of thousands of poems.

---

## 🧠 3. Model Architecture

### 🧩 Encoder v1 (Word-Level Transformer Encoder)

**Objective:** Predict the next word in a poetic sequence.

#### 🔧 Architecture Components

| Layer                     | Description                                                                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**                 | 65 tokens (64 + CLS)                                                                                                                                                |
| **Embedding**             | 144-dimensional word embeddings                                                                                                                                     |
| **Positional Encoding**   | Sinusoidal encoding added to embeddings                                                                                                                             |
| **Encoder Blocks**        | Two blocks, each with:<br>• Multi-Head Self-Attention (4 heads)<br>• Dropout + Add + LayerNorm<br>• Feed-Forward Network (4× hidden size)<br>• Residual connections |
| **CLS Vector Extraction** | Sequence summary token for prediction                                                                                                                               |
| **Feed-Forward Layer**    | Dense(256, ReLU) + Dropout                                                                                                                                          |
| **Concatenation**         | Combines CLS context and final encoder output                                                                                                                       |
| **Projection**            | Dense(144) → maps to embedding dimension                                                                                                                            |
| **Weight Tying**          | Output logits = embedding matrix^T × hidden state                                                                                                                   |
| **Softmax Output**        | Vocabulary distribution (73k tokens)                                                                                                                                |

#### ⚙️ Training Configuration

| Parameter      | Value                                                     |
| -------------- | --------------------------------------------------------- |
| **Loss**       | Sparse Categorical Crossentropy                           |
| **Optimizer**  | Adam (`lr=3e-4`)                                          |
| **Batch Size** | 1024                                                      |
| **Epochs**     | 10                                                        |
| **Callbacks**  | `EarlyStopping`, `ReduceLROnPlateau`, `HistoryCheckpoint` |
| **Accuracy**   | ~55% validation accuracy                                  |

---

## 💾 4. Custom Model Infrastructure

### 🧱 `CustomModel`

* Extends `tf.keras.Model`
* Custom **`save_weights()`** and **`load_weights()`** functions
  to save each layer as a separate `.npz` file
* Enables **resuming training** and **reloading weights** efficiently

### 📈 `HistoryCheckpoint`

* Custom callback that saves metrics after every epoch
* Persists history to JSON (e.g., `encoder_v1.history.json`)
* Allows real-time plotting or resuming of training

### 📊 `plot_history()`

Visualizes accuracy and loss curves using stored history files.

---

## 🏋️‍♂️ 5. Training Pipeline

```
train_df, val_df = split_poems(df)
train_loader = DataLoader(scanned_train_df, df, tokenizer_dict)
val_loader   = DataLoader(scanned_val_df, df, tokenizer_dict)

model = build_model_1()  # Encoder v1
model.fit(
    train_loader,
    validation_data=val_loader,
    callbacks=[EarlyStopping, ReduceLROnPlateau, HistoryCheckpoint]
)

model.save_weights("train/encoder_v1_weights/")
```

**Artifacts Generated:**

| File                              | Description                  |
| --------------------------------- | ---------------------------- |
| `/train/encoder_v1_weights/`      | Layer-by-layer model weights |
| `/train/encoder_v1.history.json`  | Training history             |
| `/train/encoder_v1_arch.png`      | Model architecture diagram   |
| `/train/plots/poem_line_dist.png` | Poem line distribution       |

---

## 🧮 6. Inference and Generation

During inference:

1. User enters a **starting prompt** via Streamlit.
2. Text is tokenized using the **same tokenizer**.
3. The prompt is fed into the model sequentially to generate the next word.
4. Generated words are appended to the prompt and re-fed for continued generation.

Streamlit ensures:

* Real-time generation
* “Generate” and “Cancel” controls
* Responsive UI design with Arabic text support

---

## 🧾 7. Comparison Summary

| Aspect             | Character-Level | Word-Level *(Implemented)* |
| ------------------ | --------------- | -------------------------- |
| Input Unit         | Characters      | Words/Subwords             |
| Convergence        | Faster          | Slower                     |
| Vocabulary Size    | Small           | Large                      |
| Semantic Coherence | Low             | High                       |
| Poetic Quality     | Weak            | Excellent                  |
| Recommended        | ❌               | ✅                          |

---

## 🧠 8. Future Work

* Implement a **Transformer Decoder** for autoregressive generation
* Add **Beam Search** and **Top-k Sampling**
* Fine-tune on specific Arabic poetry styles
* Experiment with **GPT-like** pretraining
* Integrate **attention visualization** for interpretability

---

## 👨‍💻 Author

**Youssef Samy**
Machine Learning Engineer | Benha University
Specialized in NLP & AI for creative language generation

