from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import base64
import cv2

app = Flask(__name__)
CORS(app)

# ── Comment bank from your Google Sheet ──────────────────────
COMMENTS = {
    'HASYA': {
        (0,10):   'Mokam endhuku ala pettav',
        (11,20):  'Muthi meedha mekulu kottara',
        (21,30):  'Endhuku pudutharo kuuda thelidhu',
        (31,40):  'Navvu bro koncham em kaadhu',
        (41,50):  'Parledhu serials lo act cheyochu',
        (51,60):  'Okay Movies lo side character cheyochu',
        (61,70):  'Noiceeee',
        (71,80):  'Heroooooooo',
        (81,90):  'Koncham lo national award miss ayyindhi bro',
        (91,100): 'Attttt Kamal Hassan',
    },
    'KARUNA': {
        (0,10):   'karuna chupinchali, kaamam kaadhu',
        (11,20):  'Nidra po analedhu, karuna chupinchamanam',
        (21,30):  'Kothi la pettav enti bro mokam',
        (31,40):  'Ni meedha evaraina karunisthe baagundu',
        (41,50):  'Parledhu, okay',
        (51,60):  'Noiceee, keep it up',
        (61,70):  'Acting ochu ayithe baane',
        (71,80):  'Mercy mercy mercy, ankara Mercy',
        (81,90):  'Anthe anthe ochesindhi, inkoncham',
        (91,100): 'Attttt Sai Baba',
    },
    'RAUDRA': {
        (0,10):   'Edsinatte undhi',
        (11,20):  'mokam sarey, kopam ekkada undhi',
        (21,30):  'Pilla bacha kopam idhi',
        (31,40):  'Pandu kothi la bale unnav bhaii',
        (41,50):  'kallu pedhaga chesthe kopam avvadhu nana',
        (51,60):  'Oopiri pilchuko lekapothe poye la unnav',
        (61,70):  'Eyyuuu anna',
        (71,80):  'Ammo bayam vesthundhi baboi',
        (81,90):  'Pedha actor eh',
        (91,100): 'Hey Arjun Reddy lo hero nuvve ga?',
    },
    'VEERA': {
        (0,10):   'Comedian la unnav',
        (11,20):  'Mokam enti ila undhi',
        (21,30):  'Enti ala chusthunav, ee score eh ekkuva peh',
        (31,40):  'Raju kaadhu kani, mantri ayithe okay',
        (41,50):  'Close, inkocham try cheyi',
        (51,60):  'Parledhu, okka chinna rajyam ivvochu',
        (61,70):  'Antha okay kaani edho missing king gaaru',
        (71,80):  'Abba abba em tejasuu bidda',
        (81,90):  'Meeru KGP Rajyam Prince ah?',
        (91,100): 'Raju Ekkada unna Raju eh',
    },
    'BHAYANAKA': {
        (0,10):   'Enthasepu inka act cheyadaniki',
        (11,20):  'Asalu baale',
        (21,30):  'abacha enti idhi bayame?',
        (31,40):  'Bayapettu analedhu, bayapadu annam',
        (41,50):  'Not bad, kaani inka bayam la ledhu',
        (51,60):  'Eyuuuu',
        (61,70):  'Baane bayapaduthunav',
        (71,80):  'Crush ni make-up lekunda chusava?',
        (81,90):  'Results annouce ayinattu unnayi, chaala bayapaduthunadu paapam',
        (91,100): 'Mana Main character Dhorikesar ayya',
    },
    'BIBHATSA': {
        (0,10):   'Nuvve disgusting ga unnav',
        (11,20):  'inkoncham pettochu ga expression',
        (21,30):  'inkoncham pettochu ga expression',
        (31,40):  'inkoncham pettochu ga expression',
        (41,50):  'Parledhu, okay',
        (51,60):  'Antha dharidranga undha?',
        (61,70):  'Em act chesthunav bro. Wah',
        (71,80):  'Yes idhi actor ki undalsina skill level',
        (81,90):  'Em chusav Mowa antha dhaarunanga',
        (91,100): 'Eyuuu actor',
    },
    'Adbhuta': {
        (0,10):   'Chi',
        (11,20):  'Adbhutanga cheyi annam, asahyanga kaadhu',
        (21,30):  'idhi acting ah?',
        (31,40):  'Endhuku intha lazy ga unnav',
        (41,50):  'Koncham expression kuuda pettalsindhi',
        (51,60):  'Parledhu, okay',
        (61,70):  'Anni subjects pass ayipoyava',
        (71,80):  'Crush ni saree lo chusina moment',
        (81,90):  'Chaala Adbhutanga undhi Chowdharaa',
        (91,100): 'WOWwww Noiceee',
    },
    'SHANTA': {
        (0,10):   'Yukkkkk',
        (11,20):  'Shantanga ekkada unnav?',
        (21,30):  'Enti idhi peaceful ah?',
        (31,40):  'Asale baaledhu',
        (41,50):  'Idhi eh ekkuva peh',
        (51,60):  'Peace',
        (61,70):  'Wars ni aapesela unnav ga',
        (71,80):  'Ah chiru navvu chudu eyuuu',
        (81,90):  'Gandhi jayanti ni birthday roju eh na?',
        (91,100): 'Bhudhudi la bale shantanga unnav ayya',
    },
    'SHRINGARA': {
        (0,10):   'blehhh ewww',
        (11,20):  'Enti idhi, ah maaku enti idhi antunna',
        (21,30):  'Chi',
        (31,40):  'kastame bro ila ayithe partner raavadam',
        (41,50):  'Ela padutharu anukuntunav ila evarraina',
        (51,60):  'Ayya baboiiii siguuuu ehhhhh',
        (61,70):  'ey ey eyyyyyyy',
        (71,80):  'Edho anukunamu kaani andi, maamulu vaaru kaadhandi',
        (81,90):  'Ahaaaannnn',
        (91,100): 'Rasikudive',
    },
}

NAVARASA_TO_EMOTION = {
    'HASYA':     'happy',
    'KARUNA':    'sad',
    'RAUDRA':    'angry',
    'BHAYANAKA': 'fear',
    'Adbhuta':   'surprise',
    'BIBHATSA':  'disgust',
    'SHANTA':    'neutral',
    'SHRINGARA': 'happy',
    'VEERA':     'angry',
}

def get_comment(navarasa, score):
    ranges = COMMENTS.get(navarasa, {})
    for (low, high), comment in ranges.items():
        if low <= score <= high:
            return comment
    return 'Try again!'

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'Navarasa AI API is running!'})

@app.route('/api/judge', methods=['POST'])
def judge():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        navarasa = data.get('navarasa', '')
        image_b64 = data.get('image', '')

        if not navarasa or not image_b64:
            return jsonify({'error': 'Missing navarasa or image'}), 400

        if navarasa not in NAVARASA_TO_EMOTION:
            return jsonify({'error': f'Unknown navarasa: {navarasa}'}), 400

        # Decode base64 image
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Run DeepFace
        result = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            enforce_detection=False
        )

        emotions = result[0]['emotion']
        dominant = result[0]['dominant_emotion']

        # Get score for target navarasa
        base_emotion = NAVARASA_TO_EMOTION[navarasa]
        score = round(emotions.get(base_emotion, 0))
        score = max(0, min(100, score))

        comment = get_comment(navarasa, score)

        return jsonify({
            'score': score,
            'comment': comment,
            'dominant_emotion': dominant,
            'emotions': emotions
        })

    except Exception as e:
        return jsonify({'error': str(e), 'score': 0, 'comment': 'Chi, face detect avvaledhu!'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
