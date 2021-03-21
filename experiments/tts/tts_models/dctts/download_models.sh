sudo apt update
sudo apt install gcc g++ -y
sudo apt install espeak unzip -y

git clone https://github.com/Edresson/TTS-Conv.git

mv TTS-Conv TTS_Conv

wget -c -q --show-progress -O ./saver-text.zip https://www.dropbox.com/s/oeafuy4yp7nqj5y/saver-text.zip?dl=0
unzip saver-text.zip
