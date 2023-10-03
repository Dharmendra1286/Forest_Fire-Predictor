from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        int_features = []
        for value in request.form.values():
            if value.strip():
                try:
                    int_value = int(value)
                    int_features.append(int_value)
                except ValueError:
                    pass
        if len(int_features) < 3:
            return render_template('forest_fire.html', error="Please provide at least three valid values")
        final = [np.array(int_features)]
        prediction = model.predict_proba(final)
        output = prediction[0][1]
        if output > 0.5:
            return render_template('forest_fire.html', pred=f'Your Forest is in Danger.\nProbability of fire occurring is {output:.2f}')
        else:
            return render_template('forest_fire.html', pred1=f'Your Forest is safe.\nProbability of fire occurring is {output:.2f}')
    return render_template('forest_fire.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
