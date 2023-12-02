import flask
from flask import render_template, send_from_directory
from flask_cors import CORS, cross_origin
import os

template_dir = os.path.abspath('/Users/dwijen/Documents/CODE/f2022-s2023-Battelle_NLP/spacyrenders')

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def index():
    return render_template('index.html')


@app.route("/spacyrenders/<filename>")
@cross_origin()
def get_file(filename):
    return send_from_directory(template_dir, filename)



if __name__ == "__main__":
    app.run(debug=True, port=8888)
