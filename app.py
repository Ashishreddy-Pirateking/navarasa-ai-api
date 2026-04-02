from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import os

app = Flask(**name**)
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
'HASYA': {0:'Mokam endhuku ala pettav',11:'Muthi meedha mekulu kottara',21:'Endhuku pudutharo kuuda thelidhu',31:'Navvu bro koncham em kaadhu',41:'Parledhu serials lo act cheyochu',51:'Okay Movies lo side character cheyochu',61:'Noiceeee',71:'Heroooooooo',81:'Koncham lo national award miss ayyindhi bro',91:'Attttt Kamal Hassan'},
'KARUNA': {0:'karuna chupinchali, kaamam kaadhu',11:'Nidra po analedhu, karuna chupinchamanam',21:'Kothi la pettav enti bro mokam',31:'Ni meedha evaraina karunisthe baagundu',41:'Parledhu, okay',51:'Noiceee, keep it up',61:'Acting ochu ayithe baane',71:'Mercy mercy mercy, ankara Mercy',81:'Anthe anthe ochesindhi, inkoncham',91:'Attttt Sai Baba'},
'RAUDRA': {0:'Edsinatte undhi',11:'mokam sarey, kopam ekkada undhi',21:'Pilla bacha kopam idhi',31:'Pandu kothi la bale unnav bhaii',41:'kallu pedhaga chesthe kopam avvadhu nana',51:'Oopiri pilchuko lekapothe poye la unnav',61:'Eyyuuu anna',71:'Ammo bayam vesthundhi baboi',81:'Pedha actor eh',91:'Hey Arjun Reddy lo hero nuvve ga?'},
'VEERA': {0:'Comedian la unnav',11:'Mokam enti ila undhi',21:'Enti ala chusthunav, ee score eh ekkuva peh',31:'Raju kaadhu kani, mantri ayithe okay',41:'Close, inkocham try cheyi',51:'Parledhu, okka chinna rajyam ivvochu',61:'Antha okay kaani edho missing king gaaru',71:'Abba abba em tejasuu bidda',81:'Meeru KGP Rajyam Prince ah?',91:'Raju Ekkada unna Raju eh'},
'BHAYANAKA': {0:'Enthasepu inka act cheyadaniki',11:'Asalu baale',21:'abacha enti idhi bayame?',31:'Bayapettu analedhu, bayapadu annam',41:'Not bad, kaani inka bayam la ledhu',51:'Eyuuuu',61:'Baane bayapaduthunav',71:'Crush ni make-up lekunda chusava?',81:'Results annouce ayinattu unnayi, chaala bayapaduthunadu paapam',91:'Mana Main character Dhorikesar ayya'},
'BIBHATSA': {0:'Nuvve disgusting ga unnav',11:'inkoncham pettochu ga expression',21:'inkoncham pettochu ga expression',31:'inkoncham pettochu ga expression',41:'Parledhu, okay',51:'Antha dharidranga undha?',61:'Em act chesthunav bro. Wah',71:'Yes idhi actor ki undalsina skill level',81:'Em chusav Mowa antha dhaarunanga',91:'Eyuuu actor'},
'ADBHUTA': {0:'Chi',11:'Adbhutanga cheyi annam, asahyanga kaadhu',21:'idhi acting ah?',31:'Endhuku intha lazy ga unnav',41:'Koncham expression kuuda pettalsindhi',51:'Parledhu, okay',61:'Anni subjects pass ayipoyava',71:'Crush ni saree lo chusina moment',81:'Chaala Adbhutanga undhi Chowdharaa',91:'WOWwww Noiceee'},
'SHANTA': {0:'Yukkkkk',11:'Shantanga ekkada unnav?',21:'Enti idhi peaceful ah?',31:'Asale baaledhu',41:'Idhi eh ekkuva peh',51:'Peace',61:'Wars ni aapesela unnav ga',71:'Ah chiru navvu chudu eyuuu',81:'Gandhi jayanti ni birthday roju eh na?',91:'Bhudhudi la bale shantanga unnav ayya'},
'SHRINGARA': {0:'blehhh ewww',11:'Enti idhi, ah maaku enti idhi antunna',21:'Chi',31:'kastame bro ila ayithe partner raavadam',41:'Ela padutharu anukuntunav ila evarraina',51:'Ayya baboiiii siguuuu ehhhhh',61:'ey ey eyyyyyyy',71:'Edho anukunamu kaani andi, maamulu vaaru kaadhandi',81:'Ahaaaannnn',91:'Rasikudive'},
}

def get_comment(nav, sc):
bank = COMMENTS.get(nav.upper(), COMMENTS['HASYA'])
for t in [91,81,71,61,51,41,31,21,11,0]:
if sc >= t:
return bank[t]
return bank[0]

# MODEL LOADING

print("Loading emotion model...")
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
MODEL_READY = False

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

def analyze(img, target):
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.1, 4)

```
if len(faces) == 0:
    return None

x,y,w,h = faces[0]
face = gray[y:y+h, x:x+w]
face = cv2.resize(face, (48,48))
face = face / 255.0
face = np.reshape(face, (1,48,48,1))

preds = model.predict(face, verbose=0)[0]
emotions = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}

dominant = max(emotions, key=emotions.get)
nav = FER_TO_NAVARASA.get(dominant, 'SHANTA')

target_fer = NAVARASA_TO_FER.get(target, 'neutral')
score = emotions.get(target_fer, 0)

return nav, score, emotions
```

@app.route('/')
def home():
return jsonify({'status': 'ok', 'model': MODEL_READY})

@app.route('/api/judge', methods=['POST'])
def judge():
if not MODEL_READY:
return jsonify({'error':'model not ready'})

```
data = request.get_json()
img_b64 = data['image']
nav = data['navarasa']

img_bytes = base64.b64decode(img_b64.split(',')[-1])
img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

res = analyze(img, nav)
if res is None:
    return jsonify({'score':0,'comment':'No face'})

nav_out, score, emotions = res
score = int(score*100)

return jsonify({
    'score': score,
    'comment': get_comment(nav, score),
    'dominant_emotion': nav_out,
    'emotions': emotions
})
```

if **name** == '**main**':
app.run(host='0.0.0.0', port=10000)
