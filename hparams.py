# Preprocess
cleaners = 'english_cleaners'

# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5

# Model:
hidden_size = 128
embedding_size = 256
teacher_forcing_ratio = 1.0
max_iters = 200

# Training:
outputs_per_step = 5
batch_size = 6
epochs = 100
lr = 0.001
loss_weight = 0.5
decay_step = [500000, 1000000, 2000000]
# decay_step = [20, 60]
save_step = 2000
# save_step = 20
log_step = 5
# log_step = 20
clear_Time = 100
checkpoint_path = './model_new'
