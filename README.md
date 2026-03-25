# 📘 NLU Assignment 2

## 👨‍🎓 Author
**Naman Jain (B22BB027)**  
Indian Institute of Technology Jodhpur  

---

# 🧠 PART 1: Learning Word Embeddings from IIT Jodhpur Data

## 📌 Overview

This project focuses on building **Word2Vec models (CBOW and Skip-gram)** from scratch using textual data collected from IIT Jodhpur sources. The goal is to learn meaningful word embeddings and analyze semantic relationships between academic terms.

---

## 📂 Dataset Creation

### 🔍 Data Sources

The dataset was collected from:

* IIT Jodhpur official website
* Academic regulations
* Research and academics pages
* Course-related content

### 🤖 Intelligent Data Collection Pipeline

A hybrid system was built combining:

* **Web scraping (BeautifulSoup + Requests)**
* **LLM-based filtering and cleaning (Mistral-Nemotron via NVIDIA API)**

### ⚙️ Pipeline Steps

1. Crawl IITJ webpages using BFS  
2. Extract links using BeautifulSoup  
3. Filter relevant academic links using LLM  
4. Extract meaningful text from HTML using LLM  
5. Convert structured content into paragraph format  
6. Remove noise and non-English content  

👉 Result: Clean academic corpus in paragraph form  

---

## 📊 Dataset Statistics

| Metric          | Value |
| --------------- | ----- |
| Total Tokens    | 2724  |
| Vocabulary Size | 1019  |

---

## 🧠 Models Implemented

### 1. CBOW (Continuous Bag of Words)

* Predicts target word from context  
* Faster but less expressive  

### 2. Skip-gram

* Predicts context from target word  
* Better for semantic understanding  

---

## ⚙️ Hyperparameters Explored

* Embedding Dimensions: 50, 100  
* Context Window Size: 5, 8  
* Negative Sampling: 5, 10  

---

## 🔍 Results

### 📌 Nearest Neighbors (Best Models)

#### CBOW

* Weak semantic structure for rare words  
* Good for frequent terms  

#### Skip-gram

* Strong semantic relationships  
* Captures academic hierarchy (PhD, MTech, etc.)  

---

## 🔁 Analogy Results

### CBOW

* computer - engineering + science → *performance* (0.28)  
* undergraduate - bachelor + master → *course* (0.47)  
* study - student + faculty → *skill* (0.33)  

👉 Weak relational understanding  

---

### Skip-gram

* computer - engineering + science → *technology* (0.70)  
* undergraduate - bachelor + master → *postgraduate* (0.83)  
* study - student + faculty → *application* (0.68)  

👉 Strong semantic + hierarchical relationships  

---

## 📉 Visualization

* t-SNE used for dimensionality reduction  
* Clear clustering of:  
  * Academic terms  
  * Degree programs  
  * Institutional concepts  

---

## 🧠 Key Insights

* Skip-gram significantly outperforms CBOW  
* Captures:  
  * Semantic similarity  
  * Academic hierarchy  
  * Contextual relationships  
* CBOW struggles with rare and domain-specific words  

---

## ⚠️ Limitations

* Small dataset size  
* Limited context diversity  
* Some noisy embeddings for rare words  

---

## 🚀 Future Work

* Use larger corpus  
* Compare with pre-trained embeddings (GloVe, FastText)  
* Explore transformer-based embeddings (BERT)  

---

## 📁 Project Structure (Part 1)


├── notebook.ipynb
├── Corpus.txt
└── README.md


---

## ✅ Conclusion (Part 1)

This project demonstrates how word embeddings can be learned from domain-specific data. The use of a hybrid scraping + LLM pipeline significantly improves corpus quality, leading to better semantic representations, especially with the Skip-gram model.

---

## ⭐ Note

This project was implemented from scratch without using pre-built Word2Vec libraries, ensuring a deep understanding of embedding learning mechanisms.

---

# 🔤 PART 2: Character-Level Name Generation

## 📌 Overview

This project explores **character-level sequence generation** using different recurrent neural network architectures. The goal is to generate realistic Indian names by modeling sequential dependencies between characters.

We implement and compare:

* Vanilla Recurrent Neural Network (RNN)  
* Bidirectional LSTM (BiLSTM)  
* Attention-based RNN  

---

## 🧠 Objective

The objective of this work is to learn a probabilistic model over character sequences for the task of name generation.  

Formally, given a sequence:  

x = (x₁, x₂, ..., x_T),  

the model estimates the joint probability as:  

P(x) = ∏ P(x_t | x₁, x₂, ..., x_{t-1})  

This autoregressive formulation allows the model to generate new names sequentially, by sampling each character conditioned on the preceding context.

---

## 📂 Project Structure (Part 2)


.
├── rnn_model.py
├── bilstm_model.py
├── attention_rnn.py
├── train.py
├── generate.py
├── evaluate.py
├── TrainingNames.txt
├── results.txt
├── *.pth (saved models)
└── README.md


---

## ⚙️ Models Implemented

### 1. Vanilla RNN

* Simple recurrent architecture  
* Learns sequential dependencies  
* Serves as baseline model  

---

### 2. BiLSTM

* Processes sequence in both directions  
* Uses past + future context  
* ❗ Not suitable for generation (explained below)  

---

### 3. Attention RNN

* Computes context vector using attention  
* Improves long-range dependency modeling  
* Produces best results  

---

## 🏋️ Training Details

* Framework: PyTorch  
* Loss Function: CrossEntropyLoss  
* Optimizer: Adam  
* Learning Rate: 0.002  
* Epochs: 25  

Each name is processed as:

* Input: `<name`  
* Target: `name>`  

---

## 🔤 Generation Strategy

* Start with token `<`  
* Predict next character using softmax  
* Use temperature scaling for randomness  
* Stop when `>` is generated  

---

## 📊 Results

### 🔢 Quantitative Metrics

| Model     | Novelty | Diversity |
| --------- | ------- | --------- |
| RNN       | 0.462   | 0.678     |
| BiLSTM    | 1.000   | 0.001     |
| Attention | 0.503   | 0.567     |

---

### ✨ Sample Outputs

#### RNN


anika menon
aarav pandey
myra iyer
diya singh


#### BiLSTM


(empty or repetitive outputs)


#### Attention RNN


pari malhotra
aarav bajaj
raghav bajaj
myra bajpai


---

## ❗ Key Findings

### 🔴 BiLSTM Failure

BiLSTM performs poorly for generation because:

The poor performance of BiLSTM in sequence generation arises from a fundamental mismatch between its training objective and inference setting.

During training, BiLSTM leverages bidirectional context and models:  
P(x_t | x_{<t}, x_{>t})  

In contrast, autoregressive generation requires predicting tokens based only on past context:  
P(x_t | x_{<t})  

Since future context is unavailable at inference time, the learned representations become inconsistent, leading to degraded performance, including premature termination and mode collapse.

This mismatch causes:

* Empty outputs  
* Mode collapse  
* Extremely low diversity  

---

### 🟢 Best Model: Attention RNN

* Produces coherent names  
* Maintains diversity  
* Captures long-range dependencies  

---

## ⚠️ Failure Modes

* Mode collapse (BiLSTM)  
* Premature sequence termination  
* Repetition of common surnames  
* Overfitting due to small dataset  

---

## 🚀 Future Improvements

* Use embedding layers instead of one-hot encoding  
* Add dropout regularization  
* Train on larger datasets  
* Explore Transformer-based models  

---

## ▶️ How to Run

### 1. Train Model

```bash
1. Train models
python train.py
2. Generate Names
python generate.py
3. Evaluate Models
python evaluate.py

📚 Dependencies
PyTorch
NumPy

Install dependencies: pip install torch numpy
👨‍🎓 Author

Naman Jain
B22BB027
IIT Jodhpur

📎 Note

This project was developed as part of the Natural Language Understanding (NLU) course assignment.
- Add **badges (GitHub style)**
- Add **images (t-SNE plots, results)**
- Make it **top-tier resume project README**

Just tell 👍
