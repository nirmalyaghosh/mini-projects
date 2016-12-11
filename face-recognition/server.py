import datetime
import flask
import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from facerecognition import extract_faces, predict


ALLOWED_EXTENSIONS = set(["png","jpg"])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "/home/vagrant/notebooks/data/uploads"


# Pre-trained models
model_file_path = "model_1_20_epochs.json"
model_weights_file_path = "model_1_20_epochs_weights.h5"
model = predict.load_model(model_file_path, model_weights_file_path)
names = next(os.walk("data/train"))[1]

# Used for appending timestamp
dt1970 = datetime.datetime(1970, 1, 1)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def append_timestamp(filename):
    delta = datetime.datetime.now() - dt1970
    ts_str = str(int(delta.total_seconds()*1000))
    index = filename.rfind(".")
    return "{}_{}.{}".format(filename[0:index], ts_str, filename[index+1:])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = append_timestamp(secure_filename(file.filename))
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            pred = predict.predict_person_in_photo(filepath, model, names)
            return flask.jsonify(pred)

    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    """


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
