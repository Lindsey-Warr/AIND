import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for x in range(0,len(series)-window_size):
        X.append(series[x:x+window_size])
    y = series[window_size:]
                       
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1), activation='tanh'))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    text = text
    import string
    remove_chars = []
    punctuation = ['!', ',', '.', ':', ';', '?']
    for char in set(text):
        if char not in string.ascii_lowercase and char not in string.ascii_uppercase and char not in punctuation:
            remove_chars.append(char)
        for c in set(remove_chars):
            text = text.replace(c,' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for x in range(0, (len(text)-window_size), step_size):
#         print(x)
        inputs.append(text[x:x+window_size])
        outputs.append(text[x+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation ='softmax'))
    return model
