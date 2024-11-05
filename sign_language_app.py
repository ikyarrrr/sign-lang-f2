import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load the pre-trained model
model = load_model('sign_language_model.h5')

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Define a mapping for recognized words
word_mapping = {0: "hello", 1: "thank you", 2: "please", 3: "sorry", 4: "help"}
recognized_sentence = []

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                landmark_array = np.expand_dims(landmark_array, axis=0)

                prediction = model.predict(landmark_array)
                predicted_class = np.argmax(prediction, axis=1)[0]

                recognized_word = word_mapping.get(predicted_class, "unknown")
                recognized_sentence.append(recognized_word)

                # Display recognized word
                cv2.putText(frame, recognized_word, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the recognized sentence
        sentence_display = " ".join(recognized_sentence)
        cv2.putText(frame, sentence_display, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
