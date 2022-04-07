import os
import sys
import glob
import time

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

from werkzeug.utils import secure_filename

from bg_remove import BGRemove


root = os.path.split(os.path.abspath(__file__))[0]
ckpt_image = 'pretrained/modnet_photographic_portrait_matting.ckpt'
bg_remover = BGRemove(ckpt_image)

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

TEMPLATE_FOLDER = os.path.join(root, 'web_solution', 'template')
STATIC_FOLDER = os.path.join(root, 'web_solution', 'static')
STATIC_IMAGE_PATH = os.path.join(root, 'web_solution', 'static', 'images')

UPLOAD_FOLDER = os.path.join(STATIC_IMAGE_PATH, 'test_images')
RESULT_IMAGE = os.path.join(STATIC_IMAGE_PATH, 'result_images')

make_directory = [os.makedirs(path,exist_ok=True) for path in [UPLOAD_FOLDER, RESULT_IMAGE]]

app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
            static_folder=STATIC_FOLDER)
cors = CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'PrinceAPI'


HOMEPAGE = 'home.html'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'files[]' not in request.files:
            resp = jsonify({'message': 'No file part in the request'})
            resp.status_code = 400

        files = request.files.getlist('files[]')
        errors = {}
        success = False
        file = files[0]
        filename = ""

        if file and allowed_file(file.filename):
            ts = time.time()
            filename = f"{str(ts)}-{file.filename}"
            filename = secure_filename(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors['message'] = 'File type is not allowed'

        if success and errors:
            resp = jsonify({"filepath": f"{app.config['UPLOAD_FOLDER']}/{filename}", "filename": filename,
                            'message': 'Files successfully uploaded'})
            resp.status_code = 206
        if success:
            resp = jsonify({"filepath": f"{app.config['UPLOAD_FOLDER']}/{filename}", "filename": filename,
                            'message': 'Files successfully uploaded'})
            resp.status_code = 201
        else:
            resp = jsonify(errors)
            resp.status_code = 400
        resp.html = render_template(HOMEPAGE)
        return resp

    return render_template(HOMEPAGE)


@app.route('/process/<input_filename>')
def process(input_filename):
    image = os.path.join(UPLOAD_FOLDER, input_filename)
    output_filename = bg_remover.image(
        image, background=False, output=RESULT_IMAGE, save=True)

    return render_template(HOMEPAGE, input_filename=input_filename, output_filename=output_filename)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
