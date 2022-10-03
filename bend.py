from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('randomforrest2.pkl','rb'))

@ app.route('/')
def home():
    title = 'Crop Easy - Home'
    return render_template('index.html', title=title)

@ app.route('/')
def home():
    title = 'Crop Easy - Contact'
    return render_template('contact.html', title=title)

@app.route('/')
def crop_recommend():
    return render_template("crop.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    data=[np.array(int_features)]
    print(int_features)
    print(data)
   ##prediction=randomforrest2.predict_proba(data)

    prediction=model.predict(data)

    return render_template('crop.html', output='Best suitable crop based on your inputs is {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)