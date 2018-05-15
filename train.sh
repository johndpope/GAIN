# [model = GAIN-SEC.py | GAIN-GCAM.py]
python [model].py -g 0 -f 0.99 -c
python3 [model].py -g 0 -f 0.2 -r 104999 -a inference # save predicted mask to disk

# tensorboard
tensorboard --port 7777 --logdir=[model]-saver/sum