python3 SEC.py -g 0 -f 0.06
python3 SEC.py -g 0 -f 0.05 -r 104999 -a inference
python3 GAIN-SEC.py -g 0 -f 0.45
python3 SEC.py -g 0 -f 0.05 -r 104999 -a inference

# tensorboard
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
tensorboard --port 7778 --logdir=sec-saver/sum
tensorboard --port 7779 --logdir=gain-saver/sum
