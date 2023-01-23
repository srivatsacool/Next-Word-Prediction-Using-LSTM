import streamlit as st 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from termcolor import colored, cprint
import numpy as np
import pickle
import base64
from PIL import Image
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
max_sequence_len = 40
new_model = tf.keras.models.load_model('my_model.h5')



def output_2(text,num_of_words):
    seed_text = text
    next_words = num_of_words

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        print(token_list)
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(new_model.predict(token_list), axis=-1)
        #model.predict_classes(token_list, verbose=0)
        cprint(predicted,'red')
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)
    return seed_text

#output_2('deep learning for')

import streamlit as st



st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
base="light"
st.markdown("""
            <style>
                .sidebar .sidebar-content {
                    width: 375px;
                }
                .big-font {
                    font-size:80px;
                    font-weight : 1000;
                }
                .small-font {
                    font-size:40px;
                    font-weight : 700;
                }
                .MuiGrid-item{
                    font-size:19px;
                }
                .css-1yy6isu p{
                    font-size:25px;
                }
                .st-dx{
                    font-size :18px;
                }
                .css-1fv8s86 p{
                    font-size:18px;
                }
            </style>""", unsafe_allow_html=True)

st.markdown('<p class="big-font"><center class="big-font">Next Word Prediction Using LSTM</center></p>', unsafe_allow_html=True)
st.markdown('<p class="big-font"><center class="small-font">Made by :- Srivatsa Gorti</center></p>', unsafe_allow_html=True)
st.markdown("""---""")
# st.markdown('<p> My GitHub 📖: https://github.com/srivatsacool </p>', unsafe_allow_html=True)
st.subheader("Breif Description :")
st.markdown("""
            - Based on **:red[LSTM]** model, which is a type of **:blue[RNN(Recurrent Neural Network)]**.
            - RNN is also a sub-part of DEEP LEARNING , that allow previous outputs to be used as inputs while having hidden states.
            - The datset used is  **:red[Medium Articles]** , which consists of 6508 records trained for 50 epochs.
            - Dataset Link - https://www.kaggle.com/datasets/dorianlazar/medium-articles-dataset
            - More details in **:green[GitHub]** Repo :- .
            """)  
 
st.markdown("""
<p style = 'font-size: 25px ;text-align: center'</style>Enter the Sentence you want to Auto complete</p>
""", unsafe_allow_html=True)
text = st.text_input(""," Type Here")

st.markdown("<br><br>", unsafe_allow_html=True) 

st.markdown("""
<p style = 'font-size: 25px ;text-align: center'</style>Number of words to be added/completed :-</p>
""", unsafe_allow_html=True)
level = st.slider("", 1, 20)
 
st.text(f'Selected: {level}')




if(st.button('Submit')):
    output = output_2(text,level)
    st.success(output)
    st.balloons()
    #st.snow()
    
def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
    # st.markdown(pdf_display, unsafe_allow_html=True)
    return pdf_display

col1 , col2 = st.columns(2 , gap = "small")
mark_file = show_pdf('Next Words Prediction Using Recurrent NeuralNetworks.pdf')
col1.text("Next Words Prediction Using Recurrent NeuralNetworks")
col1.markdown(mark_file, unsafe_allow_html=True)
mark_file = show_pdf('RNN and LSTM.pdf')
col2.text("What is RNN and LSTM !?")
col2.markdown(mark_file, unsafe_allow_html=True)