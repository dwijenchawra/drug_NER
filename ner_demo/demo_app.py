from flask import Flask, render_template, send_from_directory, redirect
import os
import subprocess

app = Flask(__name__)
FILE_DIR = '../data/ner_data_formatted/txt/'
OTHER_SERVER_PORT = 8888

@app.route('/')
def index():
    file_list = os.listdir(FILE_DIR)
    buttons = [{'filename': f, 'url': f'http://localhost:8888/download/{f}'} for f in file_list]
    return render_template('./index.html', buttons=buttons)

@app.route('/download/<path:filename>')
def download(filename):
    # Run some additional code here
    subprocess.run(['python', 'demo.py', filename], check=True)

    # Redirect to the other server
    return redirect(f'http://localhost:{OTHER_SERVER_PORT}/')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
