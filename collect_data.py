import cv2
import os

def collect_data(word):
    video_folder = 'dataset/' + word
    os.makedirs(video_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    print(f"Recording for {word}. Press 'r' to start and 'q' to stop.")
    
    recording = False
    video_count = len(os.listdir(video_folder))
    out = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Collecting Data', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):  # Start recording
            recording = True
            video_count += 1
            out = cv2.VideoWriter(f'{video_folder}/video_{video_count}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
            print(f"Recording {word} video {video_count}.")
        
        if recording:
            out.write(frame)
        
        if key == ord('q'):  # Stop recording
            if recording:
                recording = False
                out.release()
                print(f"Stopped recording {word}.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    words = ["hello", "thank_you", "please", "sorry", "help"]
    for word in words:
        collect_data(word)
