from flask import Flask,render_template, Response, request
import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
from text2speech import T2S
import os

model = 'attempt11_8000' 
t2s = T2S(model)
sample_text = 'Enter a sentence.'

# Initialize Flask.
app = Flask(__name__)

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    if request.method == 'POST':
        result = request.form
        model = result['input_model']
        text = result['input_text']
        max_duration_s = float(result['max_duration_s'])
        if model == t2s.model_choice and max_duration_s == t2s.max_duration_s:
            audio = t2s.tts(text)
        else:
            audio = t2s.update_model(model, max_duration_s).tts(text)
        return render_template('simple.html', voice=audio, sample_text=text, model_choice=t2s.model_choice, max_duration_s=max_duration_s)

            
#Route to render GUI
@app.route('/')
def show_entries():
    return render_template('simple.html', sample_text=sample_text, voice=None, model_choice=t2s.model_choice, max_duration_s=t2s.max_duration_s)

#Route to stream music
@app.route('/<voice>', methods=['GET'])
def streamwav(voice):
    def generate():    
        with open(os.path.join('wavs',voice), "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
            
    return Response(generate(), mimetype="audio/")

#launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 31337 
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()
    
