# Next Word Prediction Using LSTM

<p align="center">
  <img src="assets/hero.png" alt="Next Word Prediction Hero Banner" width="100%" />
</p>

## Overview

An end-to-end deep learning web application that predicts the next word a user is likely to type. Built using a **Long Short-Term Memory (LSTM)** recurrent neural network, this project demonstrates the power of sequence modeling for natural language processing tasks.

The model is trained on text corpora and learns contextual word relationships, enabling it to suggest the most probable next word given an input sequence.

---

## Live Demo

🔗 **Try it out:** [Streamlit App](https://srivatsacool-next-word-prediction-using-lstm-app-lqw5dn.streamlit.app/)

---

## Key Features

- **Real-time predictions** — Type a phrase and instantly see the predicted next word
- **LSTM architecture** — Leverages long-range dependencies in text sequences
- **Interactive UI** — Clean Streamlit interface for seamless interaction
- **Trained on real text data** — Model learns authentic language patterns

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3 | Core language |
| TensorFlow / Keras | LSTM model training and inference |
| NumPy | Numerical operations |
| Streamlit | Web application interface |
| Tokenizer | Text preprocessing and vocabulary building |

---

## How It Works

```text
User Input (text sequence)
        ↓
Tokenization & Padding
        ↓
LSTM Model Inference
        ↓
Softmax Probability Distribution
        ↓
Top Prediction Displayed
```

1. User enters a partial sentence
2. Input is tokenized and padded to match training sequence length
3. The LSTM model processes the sequence through its memory cells
4. A softmax layer outputs the probability distribution over the vocabulary
5. The word with the highest probability is displayed as the prediction

---

## Installation & Setup

```bash
git clone https://github.com/srivatsacool/Next-Word-Prediction-Using-LSTM
cd Next-Word-Prediction-Using-LSTM
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Srivatsa Gorti**

---
