from flask import Flask, render_template, request, redirect, url_for, send_file
from back import render

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/output')
def get_image():
    return send_file('output.png')

@app.route('/', methods=['POST'])
def upload_file():
	input_img = request.files['file']
	if input_img.filename != "":
		input_img.save(input_img.filename)
		fout = render(input_img.filename)
		return redirect(url_for('get_image'))
	return redirect(url_for('index'))

# @app.route('/image', methods=['GET'])
# def image():
# 	fname = request.args.get('file')
# 	# return render_template('test.html', ext= fname) 
# 	# return render_template('image.html', filename = 'output.png')
# 	return render_template('image.html')
