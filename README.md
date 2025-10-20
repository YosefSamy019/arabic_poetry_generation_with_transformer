# Arabic Poetry Generation Project 📜

This project explores the automatic generation of classical Arabic poetry using deep learning models, specifically a **Transformer Encoder** architecture configured for sequence classification (next character or next word prediction). Two distinct approaches were implemented: **Character-Level** and **Word-Level** generation.

## 1. Project Setup & Data Acquisition 🗃️

The foundational steps for both models involved setting up the environment and acquiring the necessary data.

* [cite_start]**Libraries:** Utilized Python libraries including `numpy`, `pandas`, `tensorflow` (Keras), `sklearn`, `tqdm`, and specialized Arabic processing libraries like `arabic-reshaper` and `python-bidi` for correct display[cite: 5, 6, 16, 17, 1572, 1573, 1584, 1585].
* [cite_start]**Data Source:** The project uses the **Arabic Poetry Dataset** sourced from Kaggle[cite: 2, 1569].
    * [cite_start]**Total Entries:** The dataset contains $\mathbf{54,944}$ poems[cite: 180, 1706].
    * [cite_start]**Data Columns:** The dataset has 5 columns: `id`, `category`, `poet_name`, `poem_title`, and `poem_text` [cite: 181, 184-202, 1707-1729].
    * [cite_start]**Most Frequent Category/Poet:** The most frequent category is **العصر العباسي (The Abbasid Era)**, and the most frequent poet is **ابن نباته المصري (Ibn Nubata Al-Masri)**[cite: 213, 1736, 1739, 1740].
* [cite_start]**Randomness & Constants:** A fixed `RANDOM_SEED = 42` was used for reproducibility[cite: 69, 70, 71, 1628, 1629].

***

## 2. Character-Level Model (Encoder v2 Compact Fixed)

This approach focuses on predicting the next character given a sequence of preceding characters.

### A. Data Preprocessing & Tokenization

* [cite_start]**Cleaning Function (`clean_poem_text`):** Applied comprehensive cleaning to `poem_text` and `category` columns [cite: 251-288]. This included:
    * [cite_start]Normalizing Arabic letter forms (e.g., Alif variants) [cite: 252-256].
    * [cite_start]Removing Tatweel (`ـ`) and diacritics (tashkeel) [cite: 262-265].
    * [cite_start]Removing English letters, digits, and most non-Arabic punctuation[cite: 277, 278].
    * [cite_start]Standardizing whitespace and newlines[cite: 282, 284].
* [cite_start]**Tokenization:** A character-level vocabulary was created from the unique characters in the cleaned poem texts [cite: 380-382].
    * [cite_start]**Vocabulary Size:** The tokenizer resulted in **35 tokens**[cite: 393].
    * [cite_start]**Special Tokens:** A `START_TOKEN` was used and ensured to be at index $\mathbf{0}$[cite: 63, 385, 386].
* [cite_start]**Window Generation (Scanning):** The entire poem text corpus was scanned to create fixed-size windows for the model input ($X$) and the corresponding next character for the target ($y$) [cite: 401-417].
    * [cite_start]**Window Size:** $\mathbf{WINDOW\_SIZE} = \mathbf{100}$ characters[cite: 62].
    * [cite_start]**Total Samples:** The scanning process generated **33,779,651** training samples (windows)[cite: 424].

### B. Data Splitting & Loading

* **Split Ratios:**
    * [cite_start]`TEST_SIZE` = $\mathbf{0.2}$[cite: 72].
    * [cite_start]`VAL_SIZE` = $\mathbf{0.2}$[cite: 72].
    * [cite_start]Train/Validation/Test split sizes: **$\mathbf{27,023,720}$** (Train), **$\mathbf{3,377,965}$** (Validation), **$\mathbf{3,377,966}$** (Test) samples[cite: 441, 442, 443].
* [cite_start]**Data Loader (`DataLoader`):** A custom Keras `Sequence` was implemented to efficiently load mini-batches for training[cite: 446].
    * [cite_start]**Batch Size:** $\mathbf{BATCH\_SIZE} = \mathbf{1024}$[cite: 73].
    * [cite_start]**Drop Random:** A `drop_random` of $\mathbf{0.5}$ was applied to reduce the effective number of samples per epoch, likely for faster experimentation[cite: 505].

### C. Model Architecture (Encoder v2 Compact Fixed)

The best-performing model mentioned, `Encoder v2 Compact Fixed`, is a modified Transformer Encoder designed for compactness.

* [cite_start]**Architecture Name:** `Encoder v2 Compact Fixed`[cite: 1056].
* **Core Components:**
    1.  [cite_start]**Input Layer:** Takes a sequence of length `WINDOW_SIZE` ($\mathbf{100}$)[cite: 1058, 1060].
    2.  [cite_start]**CLS Tokens:** $\mathbf{2}$ CLS tokens were prepended to the input sequence[cite: 1061, 1064, 1067].
    3.  [cite_start]**Embedding:** $\mathbf{d\_model} = \mathbf{32}$ dimensions[cite: 1057, 1069].
    4.  **Parallel Conv1D:** Multiple `Conv1D` layers with kernel sizes $\mathbf{1, 3, 5, 7}$ were concatenated to capture local context. [cite_start]The concatenated output dimension is $\mathbf{128}$ [cite: 1070-1077, 1131].
    5.  [cite_start]**Positional Encoding:** Added to the sequence embeddings [cite: 1078-1083].
    6.  [cite_start]**Multi-Head Attention:** Performed with $\mathbf{8}$ heads[cite: 1085].
    7.  [cite_start]**CLS Token Extraction:** Only the $\mathbf{2}$ CLS tokens were extracted after attention[cite: 1089, 1090].
    8.  [cite_start]**Cross Product:** A tensor cross-product was calculated on the two CLS tokens to generate a $\mathbf{(128 \times 128)}$ matrix, which was flattened to $\mathbf{16384}$ features [cite: 1092-1107].
    9.  [cite_start]**Compact Feed-Forward:** Two Dense layers ($\mathbf{256} \rightarrow \mathbf{64}$) to reduce dimensionality [cite: 1109-1111, 1133-1139].
    10. [cite_start]**Output Layer:** A Dense layer with `softmax` activation predicts the next character probability over the $\mathbf{35}$ token vocabulary[cite: 1113, 1144, 1145].
* [cite_start]**Total Parameters:** $\mathbf{4,297,283}$[cite: 1147].
* [cite_start]**Optimization:** Adam optimizer with a fixed `learning_rate` of $\mathbf{3 \times 10^{-4}}$[cite: 1118].
* [cite_start]**Callbacks:** `EarlyStopping` (patience $\mathbf{4}$), `ModelCheckpoint` for best weights, and `ReduceLROnPlateau` (patience $\mathbf{2}$) [cite: 795-806].

### D. Training Results (Encoder v2 Compact Fixed)

[cite_start]The model was trained for $\mathbf{20}$ epochs[cite: 73].

| Metric | Last Epoch (20) Train | Last Epoch (20) Validation | Best Validation |
| :--- | :--- | :--- | :--- |
| **Loss** | [cite_start]$\mathbf{1.7268}$ [cite: 1429, 1438] | [cite_start]$\mathbf{1.7789}$ [cite: 1438] | [cite_start]$\mathbf{1.77888}$ (Epoch 20) [cite: 1434, 1438] |
| **Accuracy** | [cite_start]$\mathbf{0.4992}$ [cite: 1433, 1437] | [cite_start]$\mathbf{0.4863}$ [cite: 1438] | [cite_start]$\mathbf{0.4863}$ (Epoch 20) [cite: 1438] |


### E. Inference

[cite_start]Inference was performed by iteratively predicting the next character, appending it to the prompt, and truncating the input window to maintain the fixed size [cite: 1548-1563].

* **Example Prompt (Greedy Sampling):** `"يا عبير"`
* [cite_start]**Generation Output:** Repeated and somewhat coherent Arabic phrases that cycle through similar structures (e.g., `والموت المنايا...`, `والامر المعالي...`)[cite: 1565, 1566].

***

## 3. Word-Level Model (Encoder v1)

This approach focuses on predicting the next word given a sequence of preceding words.

### A. Data Preprocessing & Tokenization

* [cite_start]**Cleaning Function (`clean_poem_text`):** Similar normalization to the Character-Level model, but crucially included steps for **morphological decomposition** to split words into sub-word units (prefixes, stems, suffixes) [cite: 1801-1867].
    * [cite_start]**Tokens for Decomposition:** $\mathbf{CONCAT\_TOKEN}$ was introduced to mark these splits[cite: 1621, 1836, 1843, 1849, 1856, 1861, 1865].
* [cite_start]**Tokenization:** A Keras `Tokenizer` was used to create the vocabulary from the pre-processed *words/sub-words*[cite: 1991].
    * [cite_start]**Vocabulary Size:** $\mathbf{73,776}$ unique tokens were loaded[cite: 2008, 2277].
    * [cite_start]**Special Tokens:** `START_TOKEN`, `CLS_TOKEN` ($\mathbf{\$}$), and $\mathbf{OOV\_TOKEN}$ were utilized[cite: 1620, 1622, 1623].
* [cite_start]**Window Generation (Scanning):** The corpus was scanned to create fixed-size windows based on *words/sub-words* [cite: 2010-2029].
    * [cite_start]**Window Size:** $\mathbf{WINDOW\_SIZE} = \mathbf{64}$ words[cite: 1619].
    * [cite_start]**Total Samples:** The scanning process generated **22,554,936** training samples[cite: 2038].

### B. Data Splitting & Loading

* **Split Ratios:**
    * [cite_start]`TEST_SIZE` = $\mathbf{0.2}$[cite: 1630].
    * [cite_start]`VAL_SIZE` = $\mathbf{0.2}$[cite: 1631].
    * [cite_start]Train/Validation/Test split sizes: **$\mathbf{18,043,948}$** (Train), **$\mathbf{2,255,494}$** (Validation), **$\mathbf{2,255,494}$** (Test) samples[cite: 2052, 2054, 2055].
* [cite_start]**Data Loader (`DataLoader`):** A custom Keras `Sequence` was used, which also prepends the `CLS_TOKEN` and pads with the `START_TOKEN`[cite: 2058, 2108].
    * [cite_start]**Batch Size:** $\mathbf{BATCH\_SIZE} = \mathbf{1024}$[cite: 1633].
    * [cite_start]**Input Sequence Length:** $\mathbf{65}$ (64 words + 1 CLS token)[cite: 2137, 2278].

### C. Model Architecture (Encoder v1)

The chosen model, `Encoder v1`, is a standard two-block Transformer Encoder.

* [cite_start]**Architecture Name:** `Encoder v1`[cite: 2276].
* **Core Components:**
    1.  [cite_start]**Input Layer:** Length $\mathbf{65}$ tokens[cite: 2278, 2280].
    2.  [cite_start]**Embedding:** $\mathbf{d\_model} = \mathbf{144}$ dimensions[cite: 2277, 2282].
    3.  [cite_start]**Positional Encoding:** Added to the sequence embeddings [cite: 2284-2287].
    4.  [cite_start]**Encoder Blocks:** $\mathbf{2}$ stacked encoder blocks [cite: 2305-2307]. Each block includes:
        * [cite_start]Multi-Head Attention (4 heads)[cite: 2291, 2292].
        * [cite_start]Add & Layer Normalization[cite: 2295, 2296].
        * [cite_start]Feed-Forward Network ($\mathbf{144} \rightarrow \mathbf{576} \rightarrow \mathbf{144}$) [cite: 2298-2301].
        * [cite_start]Add & Layer Normalization[cite: 2302, 2303].
    5.  [cite_start]**CLS Token Extraction:** Only the first token (CLS) is extracted after the encoder blocks[cite: 2309].
    6.  [cite_start]**Feed-Forward & Concatenation:** $\mathbf{144} \rightarrow \mathbf{256}$ Dense layer, followed by concatenation with the original CLS token embedding, resulting in a $\mathbf{400}$ dimension vector [cite: 2311-2313, 2333].
    7.  [cite_start]**Weight Tying Output:** A projection layer is followed by a custom Lambda layer to tie the weights to the original embedding matrix for the final output layer [cite: 2315-2319]. This is a common technique in language models.
    8.  [cite_start]**Output Layer:** `softmax` activation over the $\mathbf{73,777}$ size vocabulary[cite: 2320, 2347].
* [cite_start]**Total Parameters:** $\mathbf{11,220,160}$[cite: 2357].
* [cite_start]**Optimization:** Adam optimizer with a fixed `learning_rate` of $\mathbf{3 \times 10^{-4}}$[cite: 2325].
* [cite_start]**Callbacks:** `EarlyStopping` (patience $\mathbf{4}$) and `ReduceLROnPlateau` (patience $\mathbf{2}$)[cite: 2214, 2216].

### D. Training Results (Encoder v1)

[cite_start]The model was trained for $\mathbf{10}$ epochs[cite: 1632].

| Metric | Last Epoch (9) Train | Last Epoch (9) Validation |
| :--- | :--- | :--- |
| **Loss** | $\approx \mathbf{2.7}$ | $\approx \mathbf{2.7}$ |
| **Accuracy** | $\approx \mathbf{0.57}$ | $\approx \mathbf{0.57}$ |


### E. Inference

Two inference functions were used: `predict_txt_yield` for streaming output and `predict_txt_one_shot` for a single composed output.

* **Example Prompt (Sampling enabled):** `"يا ليل"`
* **Generation Output (Streaming):** Produces longer, more varied sequences than the character model, often including the internal concatenation tokens (`&`) due to the sub-word decomposition in the vocabulary. [cite_start]The output exhibits more diverse word choice and complex sentence structures[cite: 2526, 2527, 2528, 2529].