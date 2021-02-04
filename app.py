from flask import Flask,render_template, Response, request
import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
from text2speech import T2S
import os

language = 'en' 
t2s = T2S(language)
sample_text = {
    'en' : 'Enter the text'
}

# Initialize Flask.
app = Flask(__name__)

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    if request.method == 'POST':
        result = request.form
        lang = 'en'
        text = result['input_text']
        if lang == t2s.language:
            audio = t2s.tts(text)
        else:
            audio = t2s.update_model(lang).tts(text)
        print(audio)
        return render_template('simple.html', voice=audio, sample_text=text, opt_lang=t2s.language)

            
#Route to render GUI
@app.route('/')
def show_entries():
    return render_template('simple.html', sample_text=sample_text.get(t2s.language), voice=None, opt_lang=t2s.language)

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
    
