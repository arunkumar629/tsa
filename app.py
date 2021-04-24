from flask_cors import CORS
import os
from flask import Flask, render_template, request,jsonify
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from visualization.visualization import Visualization
from forecast_model.arima import ArimaModel
ar=ArimaModel()
draw=Visualization()
UPLOAD_FOLDER = '/data'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)
@app.route('/')
def homepage():
    return render_template('index.html')


ALLOWED_EXTENSIONS = set([ 'csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		
		file.save(os.path.join('data/', secure_filename(file.filename)))
		ar.create_model(file.filename)
		draw.chart(file.filename)
#		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		resp = jsonify({'message' : 'File successfully uploaded'})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp


if __name__ == '__main__':
    app.run(debug = True)