import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

def load_data():
    x = np.load('X.npy')
    y = np.load('y.npy')

    # Check if the arrays are empty
    if x.size == 0 or y.size == 0:
        raise ValueError("The feature array 'X' or the label array 'y' is empty. Please check the dataset.")

    return x, to_categorical(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(5, activation='softmax'))  # 5 classes for the 5 words
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    try:
        x, y = load_data()
        print(f"Loaded data with shapes: X {x.shape}, y {y.shape}")

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]))  # input shape for LSTM
        model.summary()  # Show the model architecture

        # Train the model
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
        print("Training completed.")

        # Save the model
        model.save('sign_language_model.h5')
        print("Model saved as 'sign_language_model.h5'.")

    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

