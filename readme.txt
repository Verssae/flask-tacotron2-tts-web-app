NVIDIA Tacotron2 모델로 학습된 결과물을 실행해보는 코드입니다.

실행: python tts.py <model_path> <text>

이 외에 argparse를 통해 몇 가지 인자를 줄 수 있습니다. (-h, --help 참고)

기본적으로 ./tts/plots, ./tts/wavs 에 mel output plot, 생성된 wav 파일이 저장됩니다.

패키지 설치: pip install -r requirements.txt

EXAMPLE:

python tts.py models/kr_80000 안녕하세요 반갑습니다
python tts.py models/en_90000 --lang=en Hello my name is Jonathan

python tts.py models/kr_80000 안녕하세요 반갑습니다 -r=True
plot saved at ~
audio saved at ~
> 안녕히 계세요 잘 지내세요
plot saved at ~
audio saved at ~
>

# web 에서 실행해보기
python app.py
localhost:5000


exit