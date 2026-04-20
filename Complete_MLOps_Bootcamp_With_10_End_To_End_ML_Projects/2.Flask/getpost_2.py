from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/hello', methods=['GET'])
def hello():
    name = request.args.get('name')
    return f"Hello {name} (GET)"


@app.route('/submit', methods=['POST'])
def submit():
    username = request.form.get('username')
    return f"Submitted {username} (POST)"


@app.route('/goodbye', methods=['POST'])
def goodbye():
    name = request.form.get('name')
    return f"Goodbye {name} (POST)"


if __name__ == '__main__':
    app.run(port=5003)