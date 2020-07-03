import os

from flask import Flask, render_template, request, redirect

from inference import get_prediction
from utils import generate_png

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()

        mask = get_prediction(image_bytes=img_bytes)
        mask_to_img = generate_png(img_bytes, mask)

        return render_template('result.html', mask=mask, img=mask_to_img)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
