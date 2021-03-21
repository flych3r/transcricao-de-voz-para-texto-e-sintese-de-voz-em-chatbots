# Whatsapp bot with twilio for iteraction with rasa chatbot

This bot uses [Twilio](https://www.twilio.com/) to connect to the Whapsapp api

To use it, you need to create a `twillio` account and create a project using [Whatsapp sandbox](https://www.twilio.com/console/sms/whatsapp/learn)

After the account is created, download [ngrok](https://ngrok.com/download) to allow outside connections to the Flask app.

* Run `./ngrok http 5000`
* Copy the `https` fowarding url.
* Replace the `export NGROK_URL=<ngrok forwarding https url> && python bot.py`

__*NOTE:*__ You need to add `<ngrok forwarding https url>/bot` to the twilio sandbox `WHEN A MESSAGE COMES IN` field. To test it, every user has to create a twilio account, enable Whatsapp sandbox and send a message to your sandbox first.
