# Next Word Prediction Using LSTM

## Srivatsa Gorti

---

## Abstract

Next Word Prediction is a core problem in Natural Language Processing (NLP) with applications in text autocompletion, chatbots, and intelligent typing systems.  
This project implements a **Next Word Prediction system using Long Short-Term Memory (LSTM)** networks, which are capable of learning long-term dependencies in sequential text data.

The trained deep learning model predicts the most probable next word for a given input sentence. A **Streamlit-based web application** provides real-time interaction, allowing users to input text and instantly receive predictions.

---

## Introduction

With the rapid growth of digital communication, predictive text systems have become essential for improving typing speed and user experience. Traditional statistical approaches struggle to capture long-term context in language.

LSTM networks, a variant of Recurrent Neural Networks (RNNs), overcome these limitations by maintaining internal memory states. This project explores the use of LSTM models to perform next word prediction effectively.

---

## Problem Statement

To design and implement a system that:
- Learns sequential patterns in text data
- Predicts the next word based on previous context
- Provides real-time predictions through a web interface

---

## Objectives

- Understand language modeling using LSTM
- Preprocess and tokenize textual data
- Train an LSTM-based prediction model
- Deploy the model using a web application
- Analyze system performance and limitations

---

## System Architecture

The system is divided into three primary layers:
1. **Input & Preprocessing Layer**
2. **Model Inference Layer**
3. **User Interface Layer**

### High-Level Architecture

```

User Input
↓
Text Preprocessing
(Tokenization & Padding)
↓
Trained LSTM Model
↓
Next Word Prediction
↓
Streamlit Web Interface

```

---

## Data Flow Diagram (DFD)

### DFD Level 0 (Context Diagram)

```

+-------+
| User  |
+-------+
|
| Input Sentence
v
+-----------------------------+
| Next Word Prediction System |
+-----------------------------+
|
| Predicted Word
v
+-------+
| User  |
+-------+

```

---

### DFD Level 1 (Detailed Data Flow)

```

+-------+
| User  |
+-------+
|
| Text Input
v
+---------------------------+
| Text Preprocessing        |
| - Tokenization            |
| - Sequence Padding        |
+---------------------------+
|
| Token Sequence
v
+---------------------------+
| LSTM Prediction Model     |
+---------------------------+
|
| Probability Vector
v
+---------------------------+
| Word Decoder               |
| (Index → Word Mapping)    |
+---------------------------+
|
| Predicted Next Word
v
+-------+
| User  |
+-------+

```

---

## Technology Stack

| Component | Technology |
|--------|-----------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| NLP Processing | Tokenizer, NumPy |
| Web Framework | Streamlit |
| Model Storage | Pickle |
| Version Control | Git & GitHub |

---

## Data Preparation

The text data is processed using the following steps:
- Cleaning and normalization
- Tokenization into integer sequences
- Creation of input-output word pairs
- Padding sequences to a fixed length

This ensures consistency and compatibility with the LSTM model.

---

## Model Architecture

The LSTM-based neural network includes:
- Input layer for word sequences
- Embedding layer for dense word representation
- One or more LSTM layers for sequence learning
- Dense output layer with Softmax activation

**Why LSTM?**
- Handles vanishing gradient problem
- Retains long-term contextual information
- Suitable for sequential text prediction

---

## Training and Evaluation

- Loss Function: Categorical Cross-Entropy  
- Optimizer: Adam  
- Evaluation Metric: Prediction Accuracy  

The model learns probability distributions over the vocabulary to predict the most likely next word.

---

## Web Application

The trained model is deployed using **Streamlit**, enabling:
- Real-time predictions
- Simple and interactive UI
- Instant response to user input

### Application Flow
```

User Input → Preprocessing → LSTM Model → Prediction → UI Output

````

---

## Limitations

- Limited vocabulary size
- Predicts only one word at a time
- Performance depends on training data quality

---

## Future Enhancements

- Multi-word or sentence completion
- Transformer-based models (BERT, GPT)
- Larger and more diverse datasets
- Cloud-based deployment for scalability

---

## Conclusion

This project demonstrates the successful implementation of a Next Word Prediction system using LSTM networks. The integration of deep learning with a web interface highlights the practical application of NLP techniques in real-world systems.

---

## How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
````

---

## Author

**Srivatsa Gorti**


## Demo :
Try Link - https://srivat-1--handwritten-alphanumeric-recognizer-using-cnn-1hkq8f.streamlit.app/Next_Words_Prediction_Using_LSTM


<video src="https://user-images.githubusercontent.com/76219802/214120894-e6eca151-aca1-42e1-8c18-d56aa78913f9.mp4" controls="controls" style="max-width: 1000px;" autoplay = "autoplay">
</video>


<p align="center">
  <img src="https://user-images.githubusercontent.com/76219802/214121284-51ad6243-7092-4b3a-95de-4562cff487f2.png" />
</p>

## Requirements

Install all the needed pyhton Library using requirements.txt 

```
 pip install -r requirements.txt
```
tensorflow\
termcolor\
pybase64\
pillow\
streamlit

    
## Deployment

- **IMP** * For the model file , please ask with appropriate reasons .

- Open Git Bash.

- Change the current working directory to the location where you want the cloned directory.
- Copy the URL from the repository or use the other options as well . The following is for cloning using the link .\


<p align="center">
  <img src="https://user-images.githubusercontent.com/76219802/214121375-7025e6d8-1aee-438c-951b-7f2ecea40380.png" />
</p>


- Type `git clone` , and then paste the copied URL of the repository
- Clone the repository .  

```cmd
   $ git clone https://github.com/srivatsacool/Handwritten_AlphaNumeric_Recognizer_using_CNN

To deploy this project run  :

```cmd
   streamlit run '.\app.py'
```


