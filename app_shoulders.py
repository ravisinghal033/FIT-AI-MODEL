import os
import cv2
import mediapipe as mp
import numpy as np
import threading
from playsound import playsound
from gtts import gTTS
from flask import Flask, render_template, Response
import time

app = Flask(__name__)
start_video = False

# Generate and save audio feedback
tts_up = gTTS("Up", lang="en")
tts_down = gTTS("Down", lang="en")
tts_up.save("upsh_new.mp3")
tts_down.save("downsh_new.mp3")

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

# Video streaming generator
def generate_frames():
    global max_reps, start_video
    if not start_video:
        return

    cap = cv2.VideoCapture(0)
    counter, stage = 0, None
    set=0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

                    # Extract right arm coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Extract left arm coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate shoulder angle (right shoulder)
                    right_angle = calculate_angle(right_elbow, right_shoulder, [right_shoulder[0], right_shoulder[1] - 0.1])

                    # Calculate shoulder angle (left shoulder)
                    left_angle = calculate_angle(left_elbow, left_shoulder, [left_shoulder[0], left_shoulder[1] - 0.1])

                    # Convert coordinates for display
                    right_shoulder_coords = tuple(np.multiply(right_shoulder, [640, 480]).astype(int))
                    left_shoulder_coords = tuple(np.multiply(left_shoulder, [640, 480]).astype(int))
                    right_elbow_coords = tuple(np.multiply(right_elbow, [640, 480]).astype(int))
                    left_elbow_coords = tuple(np.multiply(left_elbow, [640, 480]).astype(int))
                    right_wrist_coords = tuple(np.multiply(right_wrist, [640, 480]).astype(int))
                    left_wrist_coords = tuple(np.multiply(left_wrist, [640, 480]).astype(int))

                    # Draw circles on landmarks
                    cv2.circle(image, right_shoulder_coords, 5, (0, 255, 0), -1)  # Right Shoulder
                    cv2.circle(image, left_shoulder_coords, 5, (255, 0, 0), -1)  # Left Shoulder
                    cv2.circle(image, right_elbow_coords, 5, (0, 255, 0), -1)  # Right Elbow (green)
                    cv2.circle(image, left_elbow_coords, 5, (0, 255, 0), -1)  # Left Elbow (green)

                    # Draw lines connecting shoulder to elbow to wrist
                    cv2.line(image, right_shoulder_coords, right_elbow_coords, (255, 0, 0), 2)
                    cv2.line(image, right_elbow_coords, right_wrist_coords, (255, 0, 0), 2)
                    cv2.line(image, left_shoulder_coords, left_elbow_coords, (255, 0, 0), 2)
                    cv2.line(image, left_elbow_coords, left_wrist_coords, (255, 0, 0), 2)

                    # Display angles on screen
                    cv2.putText(image, f'Right Shoulder: {int(right_angle)}°', right_shoulder_coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f'Left Shoulder: {int(left_angle)}°', left_shoulder_coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    # Progress bar (visual indicator of movement)
                    bar_height = np.interp(right_angle, (80, 160), (100, 400))
                    cv2.rectangle(image, (20, 100), (45, 400), (255, 255, 255), 2)
                    cv2.rectangle(image, (20, int(bar_height)), (45, 400), (0, 255, 0), -1)

                    # Shoulder raise counting logic
                    if right_angle > 150 and stage != "up":
                        stage = "up"
                        play_sound("downsh_new.mp3")
                    if max_reps is not None and counter >= max_reps:
                      print("Target reps reached. Closing window.")
                      set+=1
                      time.sleep(2)
                      counter=0

                    if right_angle < 80 and stage == "up":
                        stage = "down"
                        counter += 1
                        print(f'Shoulder Raises: {counter}')
                        play_sound("upsh_new.mp3")

                # Display count and stage
                cv2.putText(image, f'Count: {counter}', (480, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Stage: {stage}', (480, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Set: {set}', (480, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 253), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error: {e}")

            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Flask routes
@app.route('/')
def index3():
    return render_template('index3.html')

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
# Run the Flask app on port 5002
if __name__ == "__main__":
    app.run(debug=True, port=5002)
