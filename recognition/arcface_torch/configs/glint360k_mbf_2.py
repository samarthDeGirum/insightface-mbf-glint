from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

config.rank_id = 1

config.margin_list = (1.0, 0.0, 0.4)
config.network = "mbf"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/data/ml-data/Glint360K/glint360k/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 10
config.warmup_epoch = 0
config.val_targets = ["lfw"]
