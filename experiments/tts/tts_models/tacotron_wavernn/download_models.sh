sudo apt update
sudo apt install gcc g++ -y
sudo apt install espeak unzip -y

git clone https://github.com/Edresson/TTS -b TTS-Portuguese
git clone https://github.com/erogol/WaveRNN.git
git -C WaveRNN checkout 12c8744

wget -c -q --show-progress -O ./TTS-TL-saver.zip https://www.dropbox.com/s/91etfwt4tvzjqyz/TTS-checkpoint-phonemizer-wavernn-381000.zip?dl=0
unzip TTS-TL-saver.zip
mv checkpoint_381000.pth.tar checkpoint.pth.tar

wget https://www.dropbox.com/s/4a60kt3detcw3r6/checkpoint-wavernn-finetunnig-tts-portuguese-corpus-560900.zip?dl=0 -O saver-wavernn.zip
unzip saver-wavernn.zip
wget https://gist.githubusercontent.com/flych3r/0f0041ac77576bb15b7032aa59b6a220/raw/ff3dd2243781ae7c4c4bff1be4f473ac75699111/wavernn.py -O WaveRNN/models/wavernn.py
mv saver.pth.tar WaveRNN/
mv config_16K.json WaveRNN/
