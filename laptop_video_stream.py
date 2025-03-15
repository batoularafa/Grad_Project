import cv2
from flask import Flask, Response

app = Flask(__name__)  # Initialize Flask app
camera = cv2.VideoCapture(0)  # Open the webcam (0 for default laptop camera)

def generate_frames():
    while True:
        success, frame = camera.read()  # Read a frame from the webcam
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)  # Convert frame to JPEG
            frame = buffer.tobytes()  # Convert to bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Stream frame

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Start the Flask server
