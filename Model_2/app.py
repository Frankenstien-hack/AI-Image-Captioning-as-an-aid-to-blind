from flask import Flask, render_template, redirect, request
import Captions

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/", methods = ["POST"])
def img():
    if request.method == "POST":
        file = request.files["userimage"]
        path = "./static/{}".format(file.filename)
        file.save(path)

        caption = Captions.caption_image(path)

        result = {
            "image": path,
            "caption": caption
        }
    return render_template("index.html", result = result)

if __name__ == "__main__":
    app.run(debug = True, threaded = False)