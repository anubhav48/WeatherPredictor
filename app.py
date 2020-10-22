import numpy as np
from flask import Flask,request, url_for, redirect, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    
    if output>str(0.5):
        return render_template('index.html',pred='You cannot ship the product tommorow.\nProbability of rain occuring is {}'.format(output))
    else:
        return render_template('index.html',pred='You can ship the product tommorow.\n Probability of rain occuring is {}'.format(output))
    

    
if __name__ == '__main__':
    app.run(debug=True)
    

