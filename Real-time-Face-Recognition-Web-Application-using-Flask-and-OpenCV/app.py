
from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition
import os
import glob

app = Flask(__name__)

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))
        for img_path in images_path:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Unable to load image {img_path}")
                    continue
                img = img.astype('uint8')  # Ensure the image is of type uint8
                # Ensure image is in RGB format
                if len(img.shape) == 2:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 3:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    print(f"Warning: Unsupported image type {img_path}")
                    continue
                basename = os.path.basename(img_path)
                (filename, ext) = os.path.splitext(basename)
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            except IndexError:
                print(f"Warning: No face found in {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Update with your image directory path

@app.route('/')
def index():
    return render_template('index.html')  # Create index.html for video display

def gen(camera):
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        face_locations, face_names = sfr.detect_known_faces(frame)
        detected = False
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            if name != 'Unknown':
                detected = True
        if detected:
            cv2.putText(frame, "Detected", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cv2.VideoCapture(0)),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
