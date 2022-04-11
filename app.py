
from flask import Flask,render_template,request
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import numpy as np
app  = Flask(__name__)
@app.route('/',methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/',methods=['POST'])
def predict(): 
    temp = request.form["message"]
    with open('tokenizer.pkl', 'rb') as input:
        tokenizer = pickle.load(input)
        Trn_seq = tokenizer.texts_to_sequences(temp)
        # print(Trn_seq)
        padding_type='post'
        truncate_type='post'
        Trn_seq = pad_sequences(Trn_seq,maxlen=50,padding=padding_type,truncating=truncate_type)
        Trn_seq = np.float32(Trn_seq)
        # Trn_seq = Trn_seq.flatten()
        print("ASDAS ",Trn_seq)
        interpreter = tf.lite.Interpreter(model_path="spamDetection.tflite")
    # Get input and output tensors.
        input_details = interpreter.get_input_details()
        print(input_details)
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], [Trn_seq])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return "iaenuci"
app.run(port = 3000, debug=True)
