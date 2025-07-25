import streamlit as st
import joblib as jl
import numpy as np
from preprocess_datasets.preprocess_dataset import embedded_text
from preprocess_datasets.preprocess_dataset import preprocessed_text
model = jl.load('phising_model.joblib')


headline = st.header("Enter Email massage : ")
user_input = st.text_input("Enter Email massage : ")
submit = st.button("Submit")
if submit and user_input != '':
  cleaned_text = preprocessed_text(user_input)
  
  embeddings = embedded_text(cleaned_text)
  embeddings_array = np.array(embeddings)
  print(f"Embeddings shape : {embeddings_array.shape}" )
  embeddings2d = np.vstack(embeddings_array) 
  print("Embedding2d shape (80) :" , embeddings2d.shape)
  
  pred = model.predict(embeddings2d)
  predict = ''
  if pred > 0.6:
    predict = 'Spam'
  else:
    predict = "Not Spam"
  st.write(f"Results : {predict}")