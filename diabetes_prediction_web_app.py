import pickle
import numpy as np
import streamlit as st

loaded_model = pickle.load(open('training_model.sav', 'rb'))


def prediction_data(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    st.title('Diabetes Prediction Web App')
