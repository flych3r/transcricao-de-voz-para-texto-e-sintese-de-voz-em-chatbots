FROM python:3.7

USER root

RUN apt-get update -qq && apt-get install -y gcc g++ ffmpeg espeak
RUN pip install --upgrade pip

RUN git clone https://github.com/flych3r/TTS -b pt-br

RUN pip install --no-cache-dir TTS/

COPY . /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5025

ADD start.sh /
RUN chmod +x /start.sh

CMD ["/start.sh"]
