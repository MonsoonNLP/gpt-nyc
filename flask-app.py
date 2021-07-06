from flask import Flask, jsonify, request
from transformers import AutoTokenizer

gpt2 = AutoTokenizer.from_pretrained('gpt2')
gptnyc = AutoTokenizer.from_pretrained('monsoon-nlp/gpt-nyc')

app = Flask(__name__)
print('ok got it')

@app.route("/")
def hello_world():
  if request.args.get('model', 'gpt2') == 'gpt2':
      model = gpt2
  else:
      model = gptnyc
  encoded = model.encode(request.args.get('txt', 'hello world'))
  tokens = list(map(lambda x: model.decode(x), encoded))
  return jsonify(tokens)
