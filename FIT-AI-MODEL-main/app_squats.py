from flask import Flask, render_template, Response
import cv2
import numpy as np
import threading
import os
from gtts import gTTS
from playsound import playsound
import mediapipe as mp

app = Flask(__name__)

# Ensure 'static' directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Generate and save audio feedback
tts_up = gTTS("Up", lang="en")
tts_down = gTTS("Down", lang="en")
tts_up.save("static/upleg_new.mp3")
tts_down.save("static/downleg_new.mp3")

# Function to calculate angles using 2D coordinates
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi
    return angle

# Function to play sound asynchronously
def play_sound(sound_file):
    threading.Thread(target=lambda: playsound(sound_file), daemon=True).start()

# Initialize MediaPipe Pose
mp_drawing, mp_pose = mp.solutions.drawing_utils, mp.solutions.pose

# Open camera
cap = cv2.VideoCapture(0)
counter, stage = 0, None

def generate_frames():
    global counter, stage
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Extract coordinates for both legs
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Check which knee is more visible
                    right_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
                    left_visibility = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility

                    if right_visibility > left_visibility:
                        hip, knee, ankle = right_hip, right_knee, right_ankle
                        side = "Right"
                    else:
                        hip, knee, ankle = left_hip, left_knee, left_ankle
                        side = "Left"

                    # Calculate knee angle
                    angle = calculate_angle(hip, knee, ankle)

                    # Convert knee coordinates for display
                    knee_coords = tuple(np.multiply(knee, [640, 480]).astype(int))
                    cv2.putText(image, f'{side} Leg: {int(angle)}°', knee_coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    # Progress bar
                    bar_height = np.interp(angle, (60, 170), (400, 100))
                    cv2.rectangle(image, (20, 100), (45, 400), (255, 255, 255), 2)
                    cv2.rectangle(image, (20, int(bar_height)), (45, 400), (0, 255, 0), -1)

                    # Squat counting logic
                    if angle > 160 and stage != "up":
                        stage = "up"
                        play_sound("static/downleg_new.mp3")

                    if angle < 110 and stage == "up":
                        stage = "down"
                        counter += 1
                        print(f'Leg Raises: {counter}')
                        play_sound("static/upleg_new.mp3")

                # Display count and stage
                cv2.putText(image, f'Count: {counter}', (480, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Stage: {stage}', (480, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
