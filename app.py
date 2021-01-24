# app.py
import json
import os
from os import walk
from flask import Flask, request
from flask_cors import CORS, cross_origin

from utils import fake_predict, fake_predict2

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/post/', methods=['POST'])
@cross_origin()
def post_data():
    comment = request.json
    print("Got comment ", comment)
    res = fake_predict(comment)  # returns fake/true
    res2 = fake_predict2(comment)  # returns clickbait/normal

    if res2 == 'clickbait' and res == 'fake':
        return {'prediction': "clickbait and probably fake"}
    if res2 == 'normal' and res == 'fake':
        return {'prediction': "probably fake"}
    if res2 == 'normal' and res == 'true':
        return {'prediction': "probably true"}
    if res2 == 'clickbait' and res == 'true':
        return {'prediction': "clickbait"}


# A welcome message to test our server
@app.route('/')
@cross_origin()
def index():
    return "Simple server for AI."


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
