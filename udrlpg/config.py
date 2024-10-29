seed = 4254148301

# gym_env_name = 'CartPole-v1'
# gym_env_name = 'InvertedPendulum-v4'
# gym_env_name = 'Swimmer-v4'
# gym_env_name = "HalfCheetah-v4"
gym_env_name = "Hopper-v4"

survival_bonus = "auto"
max_reward = "auto"
min_reward = "auto"


# command scale
max_r_scale = 1
min_r_scale = 0

# conditional variational autoencoder parameters
cvae_latent_dim = 512
kld_scaling = 1.0

# encoder parameters
enc_hidden_sizes = (256, 256, 256)
enc_normalize_out = False
enc_nlin = "leakyrelu"
enc_dropout = 0.1
enc_init_type = "xavier_uniform"  #'xavier_uniform'


# decoder (hypernetwork) parameters
dec_hidden_sizes = (512, 512)
dec_embedding_dim = 32
dec_scale_layer_out = False
dec_scale_parameter = 1


# virtual policy parameters
# input and output are fixed (input: observation space, output: action space)
virtual_policy_hidden_neurons = (256, 256)
virtual_activation_fn = "tanh"

# replay buffer parameters
buckets = True
n_buckets = 800
rbuf_max_size = 10_000
bucket_max_size = 1

# training parameters
learn_sigma = False
use_vae = False
learning_rate = 2e-6
rollout_repeats = 1
update_repeats = 8
warmup_runs = 10
batch_size = 24
update_repeats_weighting = "exp"
update_repeats_weighting_scaling = 0.3
use_obs_norm = True
moving_obs_norm = True

noise_policy = 0.1

epsilon = 1.0

warmup_from_model = True


# tracker
eval_frequency = "auto"
max_steps = "auto"
extensive_eval_frequency = "auto"
dump_buckets = False
dump_env_obs = False