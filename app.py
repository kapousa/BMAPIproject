
from flask import Flask, request
from helper.Helper import Helper

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/api/v1/<model_id>/classifydata', methods=['POST'])
def classifydata_api(model_id):
    content = request.json
    helper = Helper()
    apireturn_json = helper.classify_data(content, model_id)
    return apireturn_json


if __name__ == '__main__':
    app.run()
