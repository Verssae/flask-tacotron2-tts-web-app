FROM debian:buster

RUN apt-get update -y && apt-get install pkg-config libpng-dev libfreetype6-dev gnupg python3 python3-pip git libsndfile-dev libtinfo5 -y && apt-get autoremove -y && apt-get clean -y

RUN pip3 install --no-cache-dir matplotlib==3.1.1 numpy==1.17.5 inflect==2.1.0 librosa unidecode==1.0.22 numba==0.48 sndfile playsound==1.2.2 tornado==6.0.3 unicode==2.7 Flask==1.1.1
RUN pip3 install --no-cache-dir torch==1.7.1 torchaudio===0.7.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html 

#RUN git clone --depth 1 https://github.com/sajattack/flask-tacotron2-tts-web-app && cd flask-tacotron2-tts-web-app && git submodule update --init
COPY . /jeffotron-app
#COPY models/jeffotron_3600 /jeffotron-app/models/jeffotron_3600
#COPY models/waveglow /jeffotron-app/models/waveglow

#WORKDIR flask-tacotron2-tts-web-app
WORKDIR /jeffotron-app

ENTRYPOINT "python3" "app.py"
