import pickle
import numpy as np
import streamlit as st

loaded_model = pickle.load(open('training_model.sav', 'rb'))

st.set_page_config(page_title='Diabetes Prediction App',
                   page_icon="⚕️", initial_sidebar_state='auto')


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
    # Getting the input data from user
    Pregnancies = st.text_input('Number of Pregnancies:')
    Glucose = st.text_input('Glucose:')
    BloodPressure = st.text_input('BloodPressure:')
    SkinThickness = st.text_input('SkinThickness:')
    Insulin = st.text_input('Insulin:')
    BMI = st.text_input('BMI:')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction:')
    Age = st.text_input('Age:')

    # Code for prediction
    diagnosis = ''

    # Creating the button for prediction
    if st.button('Prediction Result'):
        input_data = [Pregnancies, Glucose, BloodPressure,
                      SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = prediction_data(input_data)

    st.success(diagnosis)


if __name__ == '__main__':
    main()
