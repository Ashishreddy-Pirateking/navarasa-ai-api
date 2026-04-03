from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import os

app = Flask(__name__)
CORS(app)

FER_TO_NAVARASA = {
'angry': 'RAUDRA',
'disgust': 'BIBHATSA',
'fear': 'BHAYANAKA',
'happy': 'HASYA',
'sad': 'KARUNA',
'surprise': 'ADBHUTA',
'neutral': 'SHANTA',
}

NAVARASA_TO_FER = {
'HASYA': 'happy', 'KARUNA': 'sad', 'RAUDRA': 'angry',
'BHAYANAKA': 'fear', 'ADBHUTA': 'surprise', 'BIBHATSA': 'disgust',
'SHANTA': 'neutral', 'SHRINGARA': 'happy', 'VEERA': 'angry',
}

COMMENTS = {
'HASYA': {0:'Mokam endhuku ala pettav',11:'Muthi meedha mekulu kottara',21:'Endhuku pudutharo kuuda thelidhu',31:'Navvu bro koncham em kaadhu',41:'Parledhu serials lo act cheyochu',51:'Okay Movies lo side character cheyochu',61:'Noiceeee',71:'Heroooooooo',81:'Koncham lo national award miss ayyindhi bro',91:'Attttt Kamal Hassan'}
}

def get_comment(nav, sc):
bank = COMMENTS.get(nav.upper(), COMMENTS['HASYA'])
for t in [91,81,71,61,51,41,31,21,11,0]:
if sc >= t:
return bank[t]
return bank[0]

print("Loading model...")
MODEL_READY = False

try:
import tensorflow as tf
import requests

```
weights_dir = "/tmp/model"
os.makedirs(weights_dir, exist_ok=True)
model_path = os.path.join(weights_dir, "model.h5")

if not os.path.exists(model_path):
    url = "https://huggingface.co/spaces/panik/Facial-Expression/resolve/2329d7eb425483a65ae56cb64550788a12401e40/facial_expression_model_weights.h5"
    r = requests.get(url, timeout=120)
    with open(model_path, "wb") as f:
        f.write(r.content)

model = tf.keras.models.load_model(model_path)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

MODEL_READY = True
print("Model ready!")
```

except Exception as e:
print("Model load error:", e)

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

def analyze(img, target):
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.1, 4)

```
if len(faces) == 0:
    return None

x, y, w, h = faces[0]
face = gray[y:y+h, x:x+w]
face = cv2.resize(face, (48,48))
face = face / 255.0
face = np.reshape(face, (1,48,48,1))

preds = model.predict(face, verbose=0)[0]
emotions = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}

dominant = max(emotions, key=emotions.get)
nav = FER_TO_NAVARASA.get(dominant, 'SHANTA')

target_fer = NAVARASA_TO_FER.get(target.upper(), 'neutral')
score = emotions.get(target_fer, 0)

return nav, score, emotions
```

@app.route("/")
def home():
return jsonify({"status": "ok", "model": "ready" if MODEL_READY else "loading"})

@app.route("/api/judge", methods=["POST"])
def judge():
if not MODEL_READY:
return jsonify({"error": "Model not ready"})

```
try:
    data = request.get_json()
    img_b64 = data["image"]
    nav = data["navarasa"]

    img_bytes = base64.b64decode(img_b64.split(",")[-1])
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    res = analyze(img, nav)

    if res is None:
        return jsonify({"score": 0, "comment": "No face detected"})

    nav_out, score, emotions = res
    score = int(score * 100)

    return jsonify({
        "score": score,
        "comment": get_comment(nav, score),
        "dominant_emotion": nav_out,
        "emotions": emotions
    })

except Exception as e:
    return jsonify({"error": str(e)})
```

if **name** == "**main**":
app.run(host="0.0.0.0", port=10000)
