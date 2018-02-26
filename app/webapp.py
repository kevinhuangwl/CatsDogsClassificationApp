from flask import Flask, request, url_for, render_template
from cnn_model import DogCatModel

app = Flask(__name__)

model = DogCatModel()

@app.route('/')
def user_interface():
	return render_template('app.html')


@app.route('/upload', methods=['POST'])
def upload_file():
	file = request.files['file']
	fname = file.filename
	path = './static/'+fname
	file.save(path)
	pred = model.predict(path)
	print(pred)
	return '{"imgPath":"'+url_for('static',filename=fname)+'","result":"'+str(pred)+'"}'


def model_predict(path):
	return model.predict(path)


if __name__ == '__main__':
	app.run()
