from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("about.html")


@app.route('/form',methods=['GET','POST'])
def form():
    if request.method == 'POST':
        pass
    return render_template("get.html")

if __name__ == '__main__':
    app.run(port=5003)