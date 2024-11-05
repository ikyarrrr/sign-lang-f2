import os
import cv2
import numpy as np
import mediapipe as mp

def process_videos(dataset_path):
    x, y = [], []
    label_map = {"hello": 0, "thank_you": 1, "please": 2, "sorry": 3, "help": 4}
    
    for label, index in label_map.items():
        video_folder = os.path.join(dataset_path, label)
        for video_file in os.listdir(video_folder):
            video_path = os.path.join(video_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            hands = mp.solutions.hands.Hands(max_num_hands=1)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame for hand detection
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        x.append(landmark_array)
                        y.append(index)
            
            cap.release()
    
    return np.array(x), np.array(y)

def save_dataset(x, y):
    np.save('X.npy', x)
    np.save('y.npy', y)

if __name__ == "__main__":
    x, y = process_videos('dataset')
    save_dataset(x, y)
    print("Dataset processed and saved.")
