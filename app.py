from flask_cors.core import serialize_option
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Predict_Text import predict_text
from tensorflow.keras.models import load_model
import pickle
import nltk

nltk.download('omw-1.4')
# app = Flask(__name__, static_folder="build/static", template_folder="build")
app = Flask(__name__)
CORS(app)

#############################################################################

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



filepath="model/sentiment_model.h5"
model1 = load_model(filepath)

with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer1= pickle.load(handle)
    
@app.route('/predict', methods=['POST'])
def predict():
#     pdb.set_trace()

    try:
        json_data = request.get_json()
        text = json_data["text"]
        print("---->",text)
    except:
         msg = {'message': "field is missing",
               'successful': False}
         return jsonify(msg)
    label, conf = predict_text([text], model1, tokenizer1)
    
    # array([0.7044847], dtype=float32)
    
    msg = {'message': label[0], 'confidence':str(round(conf[0][0]*100,2))}
    print(msg)

    return  jsonify(msg)


# # * -------------------- RUN SERVER -------------------- *
if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=False)
    
    


    
    
