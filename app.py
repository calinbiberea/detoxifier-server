# app.py
import json
import os
from os import walk
from flask import Flask, request
from flask_cors import CORS, cross_origin

from utils import fake_predict

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/post/<comment>', methods=['GET'])
@cross_origin()
def post_data(comment):
    #comment = request.args.comment
    res = fake_predict(comment)
    print(comment, res)
    return {'prediction': res}


# A welcome message to test our server
@app.route('/')
@cross_origin()
def index():
    return "Simple server for AI."


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
