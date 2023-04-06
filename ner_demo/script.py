from flask import Flask, render_template, send_from_directory, redirect
import os
import subprocess

app = Flask(__name__)
FILE_DIR = '/path/to/files'
OTHER_SERVER_PORT = 8888

@app.route('/')
def index():
    file_list = os.listdir(FILE_DIR)
    return render_template('index.html', files=file_list)

@app.route('/download/<path:filename>')
def download(filename):
    # Run some additional code here
    subprocess.run(['python', 'ner_processor.py', filename], check=True)

    # Redirect to the other server
    return redirect(f'http://localhost:{OTHER_SERVER_PORT}/{filename}')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
