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
# Generate audio files if they don't exist
if not os.path.exists("upsh_new.mp3"):
    gTTS("Up", lang="en").save("upsh_new.mp3")
if not os.path.exists("downsh_new.mp3"):
    gTTS("Down", lang="en").save("downsh_new.mp3")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180.0 / np.pi
    return angle

def play_sound(sound_file):
    threading.Thread(target=lambda: playsound(sound_file), daemon=True).start()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    def get_coords(part):
                        return [landmarks[part.value].x, landmarks[part.value].y]

                    rs, re = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER), get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
                    ls, le = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER), get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)

                    # Virtual point above shoulder
                    rs_virtual = [rs[0], rs[1] - 0.1]
                    ls_virtual = [ls[0], ls[1] - 0.1]

                    right_angle = calculate_angle(re, rs, rs_virtual)
                    left_angle = calculate_angle(le, ls, ls_virtual)

                    # Average both angles for progress
                    avg_angle = (right_angle + left_angle) / 2

                    # Convert angle to percentage (you can adjust thresholds as needed)
                    progress = np.interp(avg_angle, [60, 160], [0, 100])
                    progress = np.clip(progress, 0, 100)

                    # Count logic (based on both arms together)
                    if right_angle > 150 and left_angle > 150 and stage != "up":
                        stage = "up"
                        play_sound("downsh_new.mp3")
                    if right_angle < 80 and left_angle < 80 and stage == "up":
                        stage = "down"
                        counter += 1
                        play_sound("upsh_new.mp3")
                    if max_reps is not None and counter >= max_reps:
                      print("Target reps reached. Closing window.")
                      set+=1
                      time.sleep(2)
                      counter=0

                    # Draw progress bar
                    bar_x, bar_y = 50, 100
                    bar_height = 300
                    inverted_progress = 100 - progress
                    filled = int((inverted_progress / 100) * bar_height)
                    filled_top = bar_y + bar_height - filled  # previously this was the bottom of the green fill
                    filled_bottom = bar_y + bar_height   

                    cv2.rectangle(image, (bar_x, bar_y), (bar_x + 30, bar_y + bar_height), (0, 0, 0), 2)

                    cv2.rectangle(image, (bar_x, filled_top), (bar_x + 30, filled_bottom), (0, 255, 0), -1)
                    cv2.putText(image, f'{int(inverted_progress)}%', (bar_x - 10, bar_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Display count & stage
                    cv2.putText(image, f'Count: {counter}', (450, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(image, f'Stage: {stage}', (450, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(image, f'Set: {set}', (480, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 253), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index3():
    return render_template('index6.html')

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

if __name__ == "__main__":
    app.run(debug=True, port=5007)