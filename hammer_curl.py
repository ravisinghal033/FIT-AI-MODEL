# Save as hammer_curl.py
from flask import Flask, render_template, Response
import os , time
import cv2
import mediapipe as mp
import numpy as np
import threading
from playsound import playsound
from gtts import gTTS

app = Flask(__name__)

# Audio
tts_up = gTTS("Up", lang="en")
tts_down = gTTS("Down", lang="en")
tts_up.save("upsh_hammer.mp3")
tts_down.save("downsh_hammer.mp3")

# Angle function
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi
    return angle

# Play sound in background
def play_sound(sound_file):
    threading.Thread(target=lambda: playsound(sound_file), daemon=True).start()

# Pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def generate_frames():
    cap = cv2.VideoCapture(0)
    counter, stage = 0, None
    set=0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Get Right arm points
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate elbow angle
                    angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Convert for display
                    rs = tuple(np.multiply(right_shoulder, [640, 480]).astype(int))
                    re = tuple(np.multiply(right_elbow, [640, 480]).astype(int))
                    rw = tuple(np.multiply(right_wrist, [640, 480]).astype(int))

                    # Draw landmarks
                    cv2.circle(image, rs, 5, (255, 0, 0), -1)
                    cv2.circle(image, re, 5, (0, 255, 0), -1)
                    cv2.circle(image, rw, 5, (0, 0, 255), -1)

                    # Draw lines
                    cv2.line(image, rs, re, (255, 255, 0), 2)
                    cv2.line(image, re, rw, (255, 255, 0), 2)

                    # Angle display
                    cv2.putText(image, f'Hammer Curl Angle: {int(angle)}°', re,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                    # Bar
                    bar_height = np.interp(angle, (30, 160), (100, 400))
                    cv2.rectangle(image, (20, 100), (45, 400), (255, 255, 255), 2)
                    cv2.rectangle(image, (20, int(bar_height)), (45, 400), (0, 255, 0), -1)

                    # Count logic
                    if angle > 150 and stage != "up":
                        stage = "up"
                        play_sound("upsh_hammer.mp3")

                    if angle < 30 and stage == "up":
                        stage = "down"
                        counter += 1
                        print(f'Hammer Curls: {counter}')
                        play_sound("downsh_hammer.mp3")
                        if max_reps is not None and counter >= max_reps:
                           print("Target reps reached. Closing window.")
                           set+=1
                           time.sleep(3)
                           counter=0

                    # Display count
                    cv2.putText(image, f'Count: {counter}', (480, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Stage: {stage}', (480, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(image, f'Set: {set}', (480, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 253), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error: {e}")

            cv2.imshow('Hammer Curl Tracker', image)

            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('indes4.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/start_low', methods=['POST'])
def start_low():
    global max_reps, start_video
    max_reps = 5
    start_video = True
    return '', 204
@app.route('/start_mid', methods=['POST'])
def start_mid():
    global max_reps, start_video
    max_reps = 10
    start_video = True
    return '', 204
@app.route('/start_high', methods=['POST'])
def start_high():
    global max_reps, start_video
    max_reps = 20
    start_video = True
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=5006)  # Use a different port if running alongside other app