import base64
import os

import requests
from flask import Flask, request, send_from_directory
from pydub import AudioSegment
from twilio.twiml.messaging_response import Message, MessagingResponse

app = Flask(__name__)
NGROK_URL = os.environ.get('NGROK_URL')


def read_message(req, message_id):
    message = req.get('Body', None)
    msg_type = None
    if req.get('MediaContentType0', None) == 'audio/ogg':
        audio_msg = requests.get(req.get('MediaUrl0', ''))
        with open('input/{}.ogg'.format(message_id), 'wb') as f:
            f.write(audio_msg.content)

        with open('input/{}.ogg'.format(message_id), 'rb') as f:
            audio = f.read()
            encoded_bytes = base64.b64encode(audio)
            message = encoded_bytes.decode()
            msg_type = 'audio'
    elif message:
        msg_type = 'text'
    return message, msg_type


def send_message(resp, message_id):
    text = resp['text']
    audio = None

    if 'audio' in resp:
        with open('output/{}.ogg'.format(message_id), 'wb') as f:
            decode_string = base64.b64decode(resp['audio'])
            f.write(decode_string)
        AudioSegment.from_ogg('output/{}.ogg'.format(message_id)).export(
            'output/{}.mp3'.format(message_id), format='mp3'
        )
        audio = 'output/{}.mp3'.format(message_id)

    return text, audio


@app.route('/bot', methods=['POST'])
def bot():
    req = request.values
    sender = request.values.get('From', '')
    message_id = request.values.get('SmsSid', '')

    resp = MessagingResponse()
    msg_txt = Message()
    msg_audio = Message()

    message, msg_type = read_message(req, message_id)

    response = requests.post(
        'http://localhost:5005/webhooks/rest/webhook',
        json={
            'sender': sender,
            'message': message,
            'type': msg_type
        }
    )
    bot_resp = response.json()

    text, audio = send_message(bot_resp, message_id)
    msg_txt.body(text)
    resp.append(msg_txt)
    if audio:
        msg_audio.media('{}/{}'.format(NGROK_URL, audio))
        resp.append(msg_audio)

    return str(resp)


@app.route('/output/<path:path>')
def send(path):
    return send_from_directory('output', path)


if __name__ == '__main__':
    app.run(debug=False)
