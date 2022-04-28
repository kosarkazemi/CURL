import torch

batch_size = 1
num_epoch = 500
valid_every = 10
checkpoint_filepath = None
inference_img_dirpath = None
training_img_dirpath = "data"
start_epoch = 0
device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"

