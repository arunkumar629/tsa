import time
from flask_cors import CORS
import os
from flask import Flask, render_template, request,jsonify
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from visualization.visualization import Visualization
from forecast_model.prophet import ProphetModel 
from forecast_model.arima import ArimaModel
from anomaly.model import anomaly
ano=anomaly()


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
		#time.sleep(3)
		draw.chart(file.filename)
#		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		resp = jsonify({'message' : 'File successfully uploaded'})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp

@app.route('/forecast/<filename>')
def forecastDetails(filename):
	return jsonify(ar.timeRange(filename))

@app.route('/prophet/<filename>')
def prophetDetails(filename):
	ph=ProphetModel()
	return jsonify(ph.execute(filename))

@app.route('/anomaly/<filename>')
def anomalyDetails(filename):
	ano.abod(filename)
	ano.cluster(filename)
	ano.cof(filename)
	ano.iforest(filename)

	return "success"

@app.route('/anomalyLoad/<filename>')
def anomalyDetailsLoad(filename):
	ano.knn(filename)
	ano.lof(filename)
	ano.svm(filename)
	ano.sod(filename)
	ano.histogram(filename)
	return "success"

if __name__ == '__main__':
    app.run(debug = True)