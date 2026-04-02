from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2

app = Flask(__name__)
CORS(app)

FER_TO_NAVARASA = {
    'angry':   'RAUDRA',
    'disgust': 'BIBHATSA',
    'fear':    'BHAYANAKA',
    'happy':   'HASYA',
    'sad':     'KARUNA',
    'surprise':'ADBHUTA',
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

print("Loading FER model...")
try:
    from fer import FER
    detector = FER(mtcnn=False)
    print("FER model ready!")
except Exception as e:
    print(f"FER load error: {e}")
    detector = None

def analyze_image(img, target_navarasa):
    result = detector.detect_emotions(img)
    if not result:
        return None
    emotions_raw = result[0]['emotions']  # values are 0.0–1.0
    dominant_fer = max(emotions_raw, key=emotions_raw.get)
    dominant_navarasa = FER_TO_NAVARASA.get(dominant_fer, 'SHANTA')
    target_fer = NAVARASA_TO_FER.get(target_navarasa.upper(), 'neutral')
    target_conf = emotions_raw.get(target_fer, 0.0)
    top_val = emotions_raw.get(dominant_fer, 0.0)
    sorted_vals = sorted(emotions_raw.values(), reverse=True)
    second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    margin = top_val - second_val
    face_quality = min(1.0, margin * 2)
    target_gap = top_val - target_conf
    target_rank = sorted(emotions_raw.values(), reverse=True).index(emotions_raw.get(target_fer, 0)) + 1
    emotions_pct = {k: round(v * 100, 1) for k, v in emotions_raw.items()}
    return {
        'dominant_navarasa': dominant_navarasa,
        'dominant_fer': dominant_fer,
        'target_conf': target_conf,
        'top_val': top_val,
        'target_rank': target_rank,
        'target_gap': target_gap,
        'margin': margin,
        'face_quality': face_quality,
        'emotions_pct': emotions_pct,
    }

@app.route('/warmup', methods=['GET'])
def warmup():
    return jsonify({'status': 'warm'})

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Navarasa AI API is running!', 'model': 'FER ready' if detector else 'FER failed'})

@app.route('/predict', methods=['POST'])
def predict():
    if detector is None:
        return jsonify({'emotion': 'NO_FACE', 'confidence': 0, 'target_confidence': 0,
                        'target_raw_confidence': 0, 'raw_confidence': 0, 'margin': 0,
                        'face_quality': 0, 'target_gap': 1, 'target_rank': 9, 'face_box': None})
    try:
        data = request.get_json()
        if not data or not data.get('image'):
            return jsonify({'emotion': 'NO_FACE', 'confidence': 0, 'target_confidence': 0,
                            'target_raw_confidence': 0, 'raw_confidence': 0, 'margin': 0,
                            'face_quality': 0, 'target_gap': 1, 'target_rank': 9, 'face_box': None})

        target_navarasa = str(data.get('targetEmotion', data.get('navarasa', 'SHANTA'))).upper()
        image_b64 = data['image']
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'emotion': 'NO_FACE', 'confidence': 0, 'target_confidence': 0,
                            'target_raw_confidence': 0, 'raw_confidence': 0, 'margin': 0,
                            'face_quality': 0, 'target_gap': 1, 'target_rank': 9, 'face_box': None})

        r = analyze_image(img, target_navarasa)
        if r is None:
            return jsonify({'emotion': 'NO_FACE', 'confidence': 0, 'target_confidence': 0,
                            'target_raw_confidence': 0, 'raw_confidence': 0, 'margin': 0,
                            'face_quality': 0, 'target_gap': 1, 'target_rank': 9, 'face_box': None})

        face_box = None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
        return jsonify({'emotion': 'ERROR', 'confidence': 0, 'target_confidence': 0,
                        'target_raw_confidence': 0, 'raw_confidence': 0, 'margin': 0,
                        'face_quality': 0, 'target_gap': 1, 'target_rank': 9,
                        'face_box': None, 'error': str(e)})

@app.route('/api/judge', methods=['POST'])
def judge():
    if detector is None:
        return jsonify({'error': 'Model not loaded', 'score': 0, 'comment': 'Model load ayyindhi kaadhuu!'}), 500
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
        if r is None:
            return jsonify({'score': 0, 'comment': 'Chi, face detect avvaledhu!',
                            'dominant_emotion': 'SHANTA', 'emotions': {}})

        score = max(0, min(100, round(r['target_conf'] * 100)))
        return jsonify({
            'score': score,
            'comment': get_comment(navarasa, score),
            'dominant_emotion': r['dominant_navarasa'],
            'emotions': r['emotions_pct'],
        })
    except Exception as e:
        return jsonify({'error': str(e), 'score': 0, 'comment': 'Chi, face detect avvaledhu!'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
