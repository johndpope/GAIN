# [model = SEC.py | GAIN-SEC.py | GAIN-GCAM]
python [model].py -g 0 -f 0.45 # training
python3 [model].py -g 0 -f 0.05 -r 104999 -a inference # save predicted mask to disk

# tensorboard
tensorboard --port 7778 --logdir=[model]-saver/sum