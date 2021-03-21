#!/bin/bash

git -C TTS pull origin pt-br
if [ ! -d TTS/models ] || [ -z $UPDATE_MODEL ]; then
	gdown --id $MODEL_LINK -O tts.zip
	unzip -o tts.zip
	mkdir TTS/models
	mv tts/* TTS/models
	rm -rf tts tts.zip
fi

uvicorn app.main:app --host 0.0.0.0 --port 5025
