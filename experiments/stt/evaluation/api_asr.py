import io
import os
import time
import json
from urllib.request import urlopen
from itertools import cycle

from google.cloud import speech
import azure.cognitiveservices.speech as speechsdk
from wit import Wit
import boto3
from dotenv import load_dotenv

load_dotenv()

GCLOUD_APPLICATION_CREDENTIALS = os.getenv('GCLOUD_APPLICATION_CREDENTIALS')
GCLOUD_SPEECH_CLIENT = speech.SpeechClient.from_service_account_json(
    GCLOUD_APPLICATION_CREDENTIALS
)
GCLOUD_RECOGNITION_CONFIG = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=48000,
    audio_channel_count=1,
    language_code="pt-BR",
)


def transcribe_gcloud(speech_file):
    """Transcribe the given audio file."""

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    response = GCLOUD_SPEECH_CLIENT.recognize(config=GCLOUD_RECOGNITION_CONFIG, audio=audio)

    text = []
    for result in response.results:
        for alternative in result.alternatives:
            text.append(alternative.transcript)
    return ' '.join(text)


AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_REGION = os.getenv('AZURE_REGION')
AZURE_SPEECH_CONFIG = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY, region=AZURE_REGION
)


def transcribe_azure(file_path):
    """performs one-shot speech recognition with input from an audio file"""

    audio_input = speechsdk.AudioConfig(filename=file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=AZURE_SPEECH_CONFIG, audio_config=audio_input, language='pt-BR'
    )

    result = speech_recognizer.recognize_once_async().get()
    return result.text


AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


def upload_file_to_s3(bucket_name, file_path, file_name, region):
    s3_client = boto3.client('s3', region_name=region)

    with open(file_path, "rb") as f:
        s3_client.upload_fileobj(f, bucket_name, file_name)


def delete_file_from_s3(bucket_name, file_name, region):
    s3 = boto3.resource('s3', region_name=region)
    obj = s3.Object(bucket_name, file_name)
    obj.delete()


def transcribe_aws(file_path):

    file_name = str(file_path).split('/')[-1]
    format = file_name.split('.')[-1]

    upload_file_to_s3(AWS_BUCKET_NAME, file_path, file_name, AWS_REGION)

    job_name = 'speech2text-{}-{}'.format(file_name, time.time())
    job_uri = 'https://s3.amazonaws.com/{}/{}'.format(AWS_BUCKET_NAME, file_name)

    transcribe = boto3.client(
        'transcribe',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat=format,
        LanguageCode='pt-BR'
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(1)

    text = None
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        response = urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        data = json.loads(response.read())
        text = data['results']['transcripts'][0]['transcript']

    return text


WIT_ACCESS_TOKENS = json.loads(os.getenv('WIT_ACCESS_TOKENS'))
WIT_CLIENTS = cycle([Wit(access_token) for access_token in WIT_ACCESS_TOKENS])


def transcribe_wit(file_path):
    client = next(WIT_CLIENTS)
    with open(file_path, 'rb') as f:
        resp = client.speech(f, {'Content-Type': 'audio/wav'})
    return resp.get('text')
