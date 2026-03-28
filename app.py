from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import os

app = Flask(__name__)
CORS(app)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

FER_TO_NAVARASA = {
    'happy': 'HASYA', 'sad': 'KARUNA', 'angry': 'RAUDRA',
    'fear': 'BHAYANAKA', 'surprise': 'ADBHUTA', 'disgust': 'BIBHATSA',
    'neutral': 'SHANTA',
}

NAVARASA_TO_FER = {
    'HASYA': 'happy', 'KARUNA': 'sad', 'RAUDRA': 'angry',
    'BHAYANAKA': 'fear', 'ADBHUTA': 'surprise', 'BIBHATSA': 'disgust',
    'SHANTA': 'neutral', 'SHRINGARA': 'happy', 'VEERA': 'angry',
}

print("Loading DeepFace emotion model...")
try:
    import os, requests as req_lib
    weights_dir = "/opt/render/.deepface/weights"
    os.makedirs(weights_dir, exist_ok=True)
    h5_path = os.path.join(weights_dir, "facial_expression_model_weights.h5")

    if not os.path.exists(h5_path) or os.path.getsize(h5_path) < 1_000_000:
        print("Downloading model weights from HuggingFace...")
        url = "https://huggingface.co/spaces/panik/Facial-Expression/resolve/2329d7eb425483a65ae56cb64550788a12401e40/facial_expression_model_weights.h5"
        r = req_lib.get(url, allow_redirects=True, timeout=120)
        with open(h5_path, 'wb') as f:
            f.write(r.content)
        size = os.path.getsize(h5_path)
        print(f"Downloaded: {size} bytes")
        if size < 1_000_000:
            raise Exception(f"Downloaded file too small: {size} bytes")

    from deepface import DeepFace
    DeepFace.analyze(
        img_path=np.zeros((48, 48, 3), dtype=np.uint8),
        actions=['emotion'],
        enforce_detection=False,
        silent=True
    )
    DEEPFACE_READY = True
    print("DeepFace model ready!")
except Exception as e:
    print(f"DeepFace init warning: {e}")
    DEEPFACE_READY = False
    print("DeepFace failed to load.")


def analyze_image(img, target_navarasa):
    from deepface import DeepFace
    result = DeepFace.analyze(
        img_path=img,
        actions=['emotion'],
        enforce_detection=False,
        silent=True
    )
    emotions_raw = result[0]['emotion']
    dominant_fer = result[0]['dominant_emotion']
    dominant_navarasa = FER_TO_NAVARASA.get(dominant_fer, 'SHANTA')

    target_fer = NAVARASA_TO_FER.get(target_navarasa.upper(), 'neutral')
    target_conf = emotions_raw.get(target_fer, 0.0) / 100.0

    sorted_vals = sorted(emotions_raw.values(), reverse=True)
    top_val = sorted_vals[0] / 100.0 if sorted_vals else 1.0
    second_val = sorted_vals[1] / 100.0 if len(sorted_vals) > 1 else 0.0
    target_rank = next((i+1 for i, v in enumerate(sorted_vals)
                       if abs(v - emotions_raw.get(target_fer, 0)) < 0.01), 9)
    target_gap = float(top_val - target_conf)
    margin = float(top_val - second_val)
    face_quality = float(min(1.0, margin * 2))

    emotions_normalized = {k: v / 100.0 for k, v in emotions_raw.items()}

    return {
        'dominant_navarasa': dominant_navarasa,
        'dominant_fer': dominant_fer,
        'target_conf': target_conf,
        'top_val': top_val,
        'target_rank': target_rank,
        'target_gap': target_gap,
        'margin': margin,
        'face_quality': face_quality,
        'emotions_pct': emotions_raw,
        'emotions_norm': emotions_normalized,
    }

@app.route('/warmup', methods=['GET'])
def warmup():
    try:
        from deepface import DeepFace
        DeepFace.analyze(
            img_path=np.zeros((48, 48, 3), dtype=np.uint8),
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        return jsonify({'status': 'warm'})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/', methods=['GET'])
def health():

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Navarasa AI API is running!', 'model': 'DeepFace ready'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'emotion': 'NO_FACE', 'confidence': 0,
                            'target_confidence': 0, 'target_raw_confidence': 0,
                            'raw_confidence': 0, 'margin': 0, 'face_quality': 0,
                            'target_gap': 1, 'target_rank': 9, 'face_box': None})

        target_navarasa = str(data.get('targetEmotion', data.get('navarasa', 'SHANTA'))).upper()
        image_b64 = data.get('image', '')

        if not image_b64:
            return jsonify({'emotion': 'NO_FACE', 'confidence': 0,
                            'target_confidence': 0, 'target_raw_confidence': 0,
                            'raw_confidence': 0, 'margin': 0, 'face_quality': 0,
                            'target_gap': 1, 'target_rank': 9, 'face_box': None})

        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'emotion': 'NO_FACE', 'confidence': 0,
                            'target_confidence': 0, 'target_raw_confidence': 0,
                            'raw_confidence': 0, 'margin': 0, 'face_quality': 0,
                            'target_gap': 1, 'target_rank': 9, 'face_box': None})

        r = analyze_image(img, target_navarasa)

        face_box = None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_box = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}

        return jsonify({
            'emotion': r['dominant_navarasa'],
            'confidence': r['top_val'],
            'raw_confidence': r['top_val'],
            'target_confidence': r['target_conf'],
            'target_raw_confidence': r['target_conf'],
            'margin': r['margin'],
            'face_quality': r['face_quality'],
            'target_gap': r['target_gap'],
            'target_rank': int(r['target_rank']),
            'face_box': face_box,
            'all_emotions': r['emotions_pct'],
        })

    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'emotion': 'ERROR', 'confidence': 0,
                        'target_confidence': 0, 'target_raw_confidence': 0,
                        'raw_confidence': 0, 'margin': 0, 'face_quality': 0,
                        'target_gap': 1, 'target_rank': 9,
                        'face_box': None, 'error': str(e)})

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

        r = analyze_image(img, navarasa)
        score = round(r['target_conf'] * 100)
        score = max(0, min(100, score))

        return jsonify({
            'score': score,
            'comment': get_comment(navarasa, score),
            'dominant_emotion': r['dominant_navarasa'],
            'emotions': r['emotions_pct'],
        })

    except Exception as e:
        return jsonify({'error': str(e), 'score': 0,
                        'comment': 'Chi, face detect avvaledhu!'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
