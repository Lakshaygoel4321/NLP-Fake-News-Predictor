from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
import streamlit as st
import pickle

with open('token_word.pkl','rb') as file:
    token_word = pickle.load(file)

model = load_model('lstm_model.h5')

st.title('Fake-News Predictor')
st.write('Enter your news to Predict is it Fake or Real')


user_input = st.text_input('Enter News')

user_input_token = token_word.texts_to_sequences([user_input])

# here using the pad_sequence for making all the paragraph should be equal
user_input_sequence = pad_sequences(user_input_token,maxlen=150)


# now insert in the model to know whethe it is fake or not

if st.button('Predict'):    

    y_pred = model.predict(user_input_sequence)
    
    if y_pred>0.5:
        st.write('The News is not Fake: means News is Real')

    else:
        st.write('The News is Fake: means News is Fake')


    