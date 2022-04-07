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

app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
            static_folder=STATIC_FOLDER)
cors = CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'PrinceAPI'


HOMEPAGE = 'home.html'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
