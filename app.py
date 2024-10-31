from flask import Flask, render_template, request
from db import Database
from predict import Predict

app = Flask(__name__)
dbo = Database()
pr = Predict()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/perform_login', methods=['post'])
def perform_login():
    email = request.form.get('Email')
    password = request.form.get('Password')

    response = dbo.search(email, password)
    print(response)

    if response:
        return render_template('predict.html')
    else:
        return render_template('login.html', message='Invalid User Details. Try Again')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/perform_registration', methods=['post'])
def perform_registration():
    name = request.form.get('Name')
    email = request.form.get('Email')
    password = request.form.get('Password')

    response = dbo.insert(name, email, password)

    if response == 1:
        return render_template('login.html', message='Registered Successfully. Login to proceed')
    else:
        return render_template('register.html', message='User Already Exists')

@app.route('/predict_Iris', methods = ['post'])
def predict_Iris():
    sl = request.form.get('sl')
    sw = request.form.get('sw')
    pl = request.form.get('pl')
    pw = request.form.get('pw')

    res = pr.predict_Iris(sl,sw,pl,pw)

    return render_template('predict.html', message=res)


app.run(debug=True)
