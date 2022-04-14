import numpy as np
import pickle
loaded_model = pickle.load(open('training_model.sav', 'rb'))

input_data = (1, 97, 66, 15, 140, 23.2, 0.487, 22)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
