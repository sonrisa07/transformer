import torch.cuda
from torch.backends import mps

device = "mps" if mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters in section 6.2 of the paper
n_encoder_layers = 6
n_decoder_layers = 6
d_model = 512
d_ffn = 2048
n_heads = 8
d_k = 64
d_v = 64
dropout = 0.1
label_smoothing = 0.1
train_steps = 300

# the other hyperparameters mentioned in the paper, along with the necessary ones
batch = 128
init_lr = 1e-5
eps = 5e-9
weight_decay = 5e-4
warmup_steps = 50
factor = 0.9
clip = 1.0

# max word numbers in every sequence
enc_seq_max_len = 256
dec_seq_max_len = 256

draw_path = "./statistic.txt"
save_path = "./model.ckpt"
last_save_path = "./last_model.ckpt"

train_en_path = "./data/train.en"
train_de_path = "./data/train.de"
valid_en_path = "./data/valid.en"
valid_de_path = "./data/valid.de"
test_en_path_1 = "./data/test_2016_flickr.en"
test_de_path_1 = "./data/test_2016_flickr.de"
test_en_path_2 = "./data/test_2017_flickr.en"
test_de_path_2 = "./data/test_2017_flickr.de"
test_en_path_3 = "./data/test_2018_flickr.en"
test_de_path_3 = "./data/test_2018_flickr.de"

