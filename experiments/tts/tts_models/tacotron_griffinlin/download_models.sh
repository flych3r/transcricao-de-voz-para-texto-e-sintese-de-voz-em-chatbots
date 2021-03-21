sudo apt update
sudo apt install gcc g++ -y
sudo apt install espeak unzip -y

git clone https://github.com/Edresson/TTS -b TTS-Portuguese

wget -c -q --show-progress -O ./TTS-TL-saver.zip https://www.dropbox.com/s/91etfwt4tvzjqyz/TTS-checkpoint-phonemizer-wavernn-381000.zip?dl=0
unzip TTS-TL-saver.zip
mv checkpoint_381000.pth.tar checkpoint.pth.tar
