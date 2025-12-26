from flask import Flask, render_template, request, redirect, url_for
import subprocess
import sys

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        return redirect(url_for("menu", name=name, age=age))
    return render_template("index.html")

@app.route("/menu")
def menu():
    name = request.args.get("name")
    age = request.args.get("age")
    return render_template("menu.html", name=name, age=age)

@app.route("/sentence")
def sentence():
    subprocess.Popen([sys.executable, "live_recognition.py"])
    return "<h2>Sentence recognition started. Check webcam window.</h2>"

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
