python3 SEC.py -g 0 -f 0.1
python3 GAIN.py -g 0 -f 0.45

# tensorboard
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
tensorboard --port 7778 --logdir=sec-saver/sum
tensorboard --port 7779 --logdir=gain-saver/sum
