from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import urllib.request
import os

app = Flask(__name__)
CORS(app)

# Download model weights on startup
WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"
WEIGHTS_PATH = "/tmp/emotion_weights.h5"

def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        print("Downloading emotion model weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print("Done!")

# Build model using pure keras (no tensorflow dependency)
def build_model():
    from keras.models import Model
    from keras.layers import (Input, Conv2D, MaxPooling2D, AveragePooling2D,
                               Flatten, Dense, Dropout)
    import h5py

    inputs = Input(shape=(48, 48, 1))
    x = Conv2D(64, (5,5), activation='relu', padding='same', name='conv2d_1')(inputs)
    x = MaxPooling2D(2,2, name='max_pooling2d_1')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_2')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_3')(x)
    x = AveragePooling2D(2,2, name='average_pooling2d_1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_4')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_5')(x)
    x = AveragePooling2D(2,2, name='average_pooling2d_2')(x)
    x = Flatten(name='flatten_1')(x)
    x = Dense(1024, activation='relu', name='dense_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)
    x = Dense(1024, activation='relu', name='dense_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    outputs = Dense(7, activation='softmax', name='dense_3')(x)
    model = Model(inputs, outputs)

    LAYERS_WITH_WEIGHTS = ['conv2d_1','conv2d_2','conv2d_3',
                            'conv2d_4','conv2d_5',
                            'dense_1','dense_2','dense_3']
    with h5py.File(WEIGHTS_PATH, 'r') as f:
        for layer_name in LAYERS_WITH_WEIGHTS:
            layer = model.get_layer(layer_name)
            kernel = f[layer_name][layer_name]['kernel:0'][:]
            bias   = f[layer_name][layer_name]['bias:0'][:]
            layer.set_weights([kernel, bias])

    print("Model loaded!")
    return model

# Load on startup
download_weights()
emotion_model = build_model()
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ── Comments ──────────────────────────────────────────────────
COMMENTS = {
    'HASYA': {(0,10):'Mokam endhuku ala pettav',(11,20):'Muthi meedha mekulu kottara',(21,30):'Endhuku pudutharo kuuda thelidhu',(31,40):'Navvu bro koncham em kaadhu',(41,50):'Parledhu serials lo act cheyochu',(51,60):'Okay Movies lo side character cheyochu',(61,70):'Noiceeee',(71,80):'Heroooooooo',(81,90):'Koncham lo national award miss ayyindhi bro',(91,100):'Attttt Kamal Hassan'},
    'KARUNA': {(0,10):'karuna chupinchali, kaamam kaadhu',(11,20):'Nidra po analedhu, karuna chupinchamanam',(21,30):'Kothi la pettav enti bro mokam',(31,40):'Ni meedha evaraina karunisthe baagundu',(41,50):'Parledhu, okay',(51,60):'Noiceee, keep it up',(61,70):'Acting ochu ayithe baane',(71,80):'Mercy mercy mercy, ankara Mercy',(81,90):'Anthe anthe ochesindhi, inkoncham',(91,100):'Attttt Sai Baba'},
    'RAUDRA': {(0,10):'Edsinatte undhi',(11,20):'mokam sarey, kopam ekkada undhi',(21,30):'Pilla bacha kopam idhi',(31,40):'Pandu kothi la bale unnav bhaii',(41,50):'kallu pedhaga chesthe kopam avvadhu nana',(51,60):'Oopiri pilchuko lekapothe poye la unnav',(61,70):'Eyyuuu anna',(71,80):'Ammo bayam vesthundhi baboi',(81,90):'Pedha actor eh',(91,100):'Hey Arjun Reddy lo hero nuvve ga?'},
    'VEERA': {(0,10):'Comedian la unnav',(11,20):'Mokam enti ila undhi',(21,30):'Enti ala chusthunav, ee score eh ekkuva peh',(31,40):'Raju kaadhu kani, mantri ayithe okay',(41,50):'Close, inkocham try cheyi',(51,60):'Parledhu, okka chinna rajyam ivvochu',(61,70):'Antha okay kaani edho missing king gaaru',(71,80):'Abba abba em tejasuu bidda',(81,90):'Meeru KGP Rajyam Prince ah?',(91,100):'Raju Ekkada unna Raju eh'},
    'BHAYANAKA': {(0,10):'Enthasepu inka act cheyadaniki',(11,20):'Asalu baale',(21,30):'abacha enti idhi bayame?',(31,40):'Bayapettu analedhu, bayapadu annam',(41,50):'Not bad, kaani inka bayam la ledhu',(51,60):'Eyuuuu',(61,70):'Baane bayapaduthunav',(71,80):'Crush ni make-up lekunda chusava?',(81,90):'Results annouce ayinattu unnayi, chaala bayapaduthunadu paapam',(91,100):'Mana Main character Dhorikesar ayya'},
    'BIBHATSA': {(0,10):'Nuvve disgusting ga unnav',(11,20):'inkoncham pettochu ga expression',(21,30):'inkoncham pettochu ga expression',(31,40):'inkoncham pettochu ga expression',(41,50):'Parledhu, okay',(51,60):'Antha dharidranga undha?',(61,70):'Em act chesthunav bro. Wah',(71,80):'Yes idhi actor ki undalsina skill level',(81,90):'Em chusav Mowa antha dhaarunanga',(91,100):'Eyuuu actor'},
    'Adbhuta': {(0,10):'Chi',(11,20):'Adbhutanga cheyi annam, asahyanga kaadhu',(21,30):'idhi acting ah?',(31,40):'Endhuku intha lazy ga unnav',(41,50):'Koncham expression kuuda pettalsindhi',(51,60):'Parledhu, okay',(61,70):'Anni subjects pass ayipoyava',(71,80):'Crush ni saree lo chusina moment',(81,90):'Chaala Adbhutanga undhi Chowdharaa',(91,100):'WOWwww Noiceee'},
    'SHANTA': {(0,10):'Yukkkkk',(11,20):'Shantanga ekkada unnav?',(21,30):'Enti idhi peaceful ah?',(31,40):'Asale baaledhu',(41,50):'Idhi eh ekkuva peh',(51,60):'Peace',(61,70):'Wars ni aapesela unnav ga',(71,80):'Ah chiru navvu chudu eyuuu',(81,90):'Gandhi jayanti ni birthday roju eh na?',(91,100):'Bhudhudi la bale shantanga unnav ayya'},
    'SHRINGARA': {(0,10):'blehhh ewww',(11,20):'Enti idhi, ah maaku enti idhi antunna',(21,30):'Chi',(31,40):'kastame bro ila ayithe partner raavadam',(41,50):'Ela padutharu anukuntunav ila evarraina',(51,60):'Ayya baboiiii siguuuu ehhhhh',(61,70):'ey ey eyyyyyyy',(71,80):'Edho anukunamu kaani andi, maamulu vaaru kaadhandi',(81,90):'Ahaaaannnn',(91,100):'Rasikudive'},
}

NAVARASA_TO_EMOTION = {
    'HASYA':'happy','KARUNA':'sad','RAUDRA':'angry',
    'BHAYANAKA':'fear','Adbhuta':'surprise','BIBHATSA':'disgust',
    'SHANTA':'neutral','SHRINGARA':'happy','VEERA':'angry',
}

def get_comment(navarasa, score):
    for (low, high), comment in COMMENTS.get(navarasa, {}).items():
        if low <= score <= high:
            return comment
    return 'Try again!'

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        gray = gray[y:y+h, x:x+w]
    gray = cv2.resize(gray, (48, 48))
    arr = gray.astype('float32') / 255.0
    return arr.reshape(1, 48, 48, 1)

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Navarasa AI API is running!'})

@app.route('/api/judge', methods=['POST'])
def judge():
    try:
        data = request.get_json()
        navarasa = data.get('navarasa', '')
        image_b64 = data.get('image', '')

        if not navarasa or not image_b64:
            return jsonify({'error': 'Missing data'}), 400

        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        face = preprocess_face(img)
        preds = emotion_model.predict(face, verbose=0)[0]

        emotions = {e: round(float(p * 100), 1) for e, p in zip(EMOTIONS, preds)}
        base_emotion = NAVARASA_TO_EMOTION.get(navarasa, 'neutral')
        score = round(emotions.get(base_emotion, 0))
        score = max(0, min(100, score))
        comment = get_comment(navarasa, score)

        return jsonify({
            'score': score,
            'comment': comment,
            'dominant_emotion': EMOTIONS[int(np.argmax(preds))],
            'emotions': emotions
        })

    except Exception as e:
        return jsonify({'error': str(e), 'score': 0, 'comment': 'Chi, face detect avvaledhu!'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
