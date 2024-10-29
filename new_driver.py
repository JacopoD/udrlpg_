from udrlpg.cvae import ConditionalVAE
from udrlpg.utils import parse_config, ReplayBuffer
from udrlpg.my_new_env import MyEnv
from udrlpg.virtualmlpp import VirtualMLPPolicy
import torch
import gymnasium as gym
import numpy as np
import random
from udrlpg.tracker import Tracker
from gymnasium.spaces import Box, Discrete
import wandb
import udrlpg.config as local_config
import argparse
from torch.distributions import Normal
from udrlpg.dynamic_buckets_fifo import ReplayBufferBucket
from udrlpg.my_hypernetwork import get_hypernetwork_mlp_generator


def train(
    model: ConditionalVAE,
    optimizer,
    device,
    replay_buffer: ReplayBufferBucket,
    my_env: MyEnv,
    tracker: Tracker,
    config,
):
    model.train()
    print("Started training...")

    should_exit = False
    while not should_exit:
        for j in range(1, config.update_repeats + 1):
            commands, weights = replay_buffer.sample(
                config.batch_size,
                weights=config.update_repeats_weighting,
                scaling=config.update_repeats_weighting_scaling,
            )

            commands = (
                torch.tensor(commands, dtype=torch.float32).unsqueeze(-1).to(device)
            )
            weights = torch.vstack(weights).to(device)

            optimizer.zero_grad()

            if config.use_vae:
                _, decoded, mean, log_var = model.forward(weights, commands)
                rec_loss, reg_loss = model.loss(
                    weights, decoded, mean, log_var, return_separately=True
                )
                loss = rec_loss + reg_loss
            else:
                decoded = model.forward(commands)
                loss = torch.nn.functional.mse_loss(decoded, weights, reduction="mean")

            tracker.loss.append(loss.item())
            
            loss.backward()
            optimizer.step()

            if config.use_vae:
                tracker.reg_loss.append(reg_loss.item())
                tracker.rec_loss.append(rec_loss.item())
            tracker.max_reward_seen = my_env.scale_from_my_to_env(
                replay_buffer.max_seen
            )
            if config.learn_sigma and config.use_vae:
                tracker.decoder_log_sigma.append(model.log_sigma.detach().item())

        for j in range(config.rollout_repeats):
            command = replay_buffer.max_seen

            command = torch.tensor([command], dtype=torch.float32).unsqueeze(-1)

            command = command.to(device)
            if config.use_vae:
                policy_w = model.sample(command).squeeze(0)
            else:
                with torch.no_grad():
                    policy_w = model.forward(command).squeeze(0)

            no_noise_r = None
            if config.epsilon == 1.0 or np.random.rand() < config.epsilon:
                with torch.no_grad():
                    no_noise_r = my_env.do_episode(
                        policy_w,
                        is_train=False,
                        scale_reward=True,
                        remove_survival_bonus=(not config.survival_bonus),
                    )

                p_dist = Normal(torch.zeros(my_env.policy.num_parameters), scale=1)
                p_delta = p_dist.sample().to(device=device, non_blocking=True).detach()
                # policy_w + N(0,standard_deviation); std=exp(log_sigma)
                if config.learn_sigma:
                    policy_w = (
                        policy_w + model.log_sigma.detach().exp().item() * p_delta
                    )
                else:
                    policy_w = policy_w + config.noise_policy * p_delta

            r = my_env.do_episode(policy_w, is_train=True, scale_reward=True)

            if no_noise_r is None:
                no_noise_r = r

            if r is None:
                should_exit = True
                break
            else:
                replay_buffer.add((r, policy_w))
                tracker.push_rollout_history(
                    before_noise=my_env.scale_from_my_to_env(no_noise_r),
                    after_noise=my_env.scale_from_my_to_env(r),
                )
                wandb.log(
                    {
                        "rollout_history_after_noise": tracker.rollout_history_after_noise[
                            -1
                        ],
                        "rollout_history_before_noise": tracker.rollout_history_before_noise[
                            -1
                        ],
                    },
                    step=my_env.global_steps,
                )

        print(f"{(my_env.global_steps*100/config.max_steps):.2f}%\t\t", end="\r")

    if config.dump_buckets:
        tracker._save_buckets_to_file(replay_buffer)
    if config.dump_env_obs:
        tracker._save_obs_history_to_file(my_env)
    tracker.done()


def main(is_sweep: bool, project_name: str, use_wandb: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_sweep:
        run = wandb.init()
        config = wandb.config

        if not hasattr(config, "seed"):
            seed = random.randrange(0, 4294967295)
        else:
            seed = config.seed

        print(seed)

        project_name = run.project_name()

    else:
        cdict = dict(
            [(a, b) for (a, b) in vars(local_config).items() if not a.startswith("__")]
        )

        config = local_config
        if config.seed is None:
            seed = random.randrange(0, 4294967295)
        else:
            seed = config.seed

        cdict["seed"] = seed

        run = wandb.init(
            project=project_name,
            config=cdict,
            mode=None if use_wandb else "disabled",
        )

    config = parse_config(config, is_sweep)

    tracker = Tracker(
        save_path=f"/home/jacopo/storage/models/{project_name}/{config.gym_env_name}",
        folder_name=f"{run.name}",
        config=config,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    train_gym_env = gym.make(config.gym_env_name)
    eval_gym_env = gym.make(config.gym_env_name)

    if config.buckets:
        replay_buffer = ReplayBufferBucket(
            n_buckets=config.n_buckets,
            rng=rng,
            max_size=config.rbuf_max_size,
            bucket_max_size=config.bucket_max_size,
        )
    else:
        replay_buffer = ReplayBuffer(
            rng=rng,
            max_size=config.rbuf_max_size,
        )

    if isinstance(train_gym_env.action_space, Discrete):
        policy_neurons = (
            [train_gym_env.observation_space.shape[0]]
            + list(config.virtual_policy_hidden_neurons)
            + [train_gym_env.action_space.n]
        )
        vmlp = VirtualMLPPolicy(
            layer_sizes=policy_neurons,
            act_lim=1,
            nonlinearity=config.virtual_activation_fn,
            output_activation="linear",
        )
    elif isinstance(train_gym_env.action_space, Box):
        policy_neurons = (
            [train_gym_env.observation_space.shape[0]]
            + list(config.virtual_policy_hidden_neurons)
            + [train_gym_env.action_space.shape[0]]
        )
        vmlp = VirtualMLPPolicy(
            layer_sizes=policy_neurons,
            act_lim=train_gym_env.action_space.high[0],
            nonlinearity=config.virtual_activation_fn,
            output_activation="tanh",
        )
    else:
        raise NotImplementedError

    if config.use_vae:
        model = ConditionalVAE(
            input_dim=vmlp.num_parameters,
            latent_dim=config.cvae_latent_dim,
            kld_scaling=config.kld_scaling,
            enc_hidden_sizes=config.enc_hidden_sizes,
            enc_normalize_out=config.enc_normalize_out,
            enc_nlin=config.enc_nlin,
            enc_dropout=config.enc_dropout,
            policy_neurons=policy_neurons,
            dec_hidden_sizes=config.dec_hidden_sizes,
            embedding_dim=config.dec_embedding_dim,
            scale_layer_out=config.dec_scale_layer_out,
            scale_parameter=config.dec_scale_parameter,
            learn_sigma=config.learn_sigma,
            device=device,
        ).to(device)
    else:
        model = get_hypernetwork_mlp_generator(
            layer_sizes=policy_neurons,
            hidden_sizes=config.dec_hidden_sizes,
            embedding_dim=config.dec_embedding_dim,
            scale_layer_out=config.dec_scale_layer_out,
            scale_parameter=config.dec_scale_parameter,
            command_len=1,
        ).to(device)

    my_env = MyEnv(
        train_env=train_gym_env,
        eval_env=eval_gym_env,
        train_seed=seed,
        eval_seed=random.randrange(0, 4294967295),
        policy=vmlp,
        device=device,
        config=config,
        model=model,
        tracker=tracker,
        rbuf=replay_buffer,
    )

    if config.use_obs_norm and (not config.moving_obs_norm):
        if config.gym_env_name == "InvertedPendulum-v4":
            my_env.std = np.array([0.36802367, 0.04843017, 0.61370338, 1.20090304])
            my_env.mean = np.array([-0.02793755, -0.00082555, 0.00847352, -0.000621])
        elif config.gym_env_name == "Swimmer-v4":
            my_env.mean = np.array(
                [
                    -0.02772265,
                    0.02953586,
                    0.07311105,
                    0.04740786,
                    -0.00382523,
                    -0.00032485,
                    0.00067412,
                    0.00162605,
                ]
            )
            my_env.std = np.array(
                [
                    1.19326452,
                    1.26283398,
                    1.32160237,
                    0.85102312,
                    1.24168655,
                    1.62174071,
                    2.52871592,
                    2.52623194,
                ]
            )
        elif config.gym_env_name == "Hopper-v4":
            my_env.std = np.array(
                [
                    0.13669864,
                    0.0756167,
                    0.24965658,
                    0.26942532,
                    0.4650226,
                    0.83210441,
                    0.91152866,
                    1.01927618,
                    1.18317537,
                    1.43168147,
                    4.30347461,
                ]
            )
            my_env.mean = np.array(
                [
                    1.24017734,
                    -0.00567254,
                    -0.19667449,
                    -0.13538874,
                    0.18951969,
                    0.70348488,
                    -0.10130716,
                    -0.06635226,
                    -0.26434784,
                    -0.28622685,
                    0.0290802,
                ]
            )
        else:
            raise NotImplementedError

    # TODO
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for _ in range(config.warmup_runs):
        if config.warmup_from_model:
            command = torch.tensor([config.max_r_scale], dtype=torch.float32).unsqueeze(
                -1
            )
            command = command.to(device)
            if config.use_vae:
                policy_w = model.sample(command).squeeze(0)
            else:
                with torch.no_grad():
                    policy_w = model.forward(command).squeeze(0)
        else:
            policy_w = vmlp._get_xavier_weights().to(device)

        r = my_env.do_episode(
            policy_w, is_train=True, scale_reward=True, disable_evaluation=True
        )
        replay_buffer.add((r, policy_w))

    train(
        model=model,
        optimizer=optimizer,
        device=device,
        replay_buffer=replay_buffer,
        my_env=my_env,
        tracker=tracker,
        config=config,
    )


if __name__ == "__main__":
    np.seterr(invalid="raise")
    wandb.login(key="xxx")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", type=str, required=False, default=None)
    parser.add_argument("--projn", type=str, required=True, default=None)
    parser.add_argument("--count", type=int, required=False, default=100)
    parser.add_argument("--nowandb", action="store_true")
    args = parser.parse_args()
    if args.sid is None:
        if args.nowandb:
            main(is_sweep=False, project_name=args.projn, use_wandb=False)
        else:
            main(is_sweep=False, project_name=args.projn, use_wandb=True)
    else:
        if args.nowandb:
            raise NotImplementedError("Can't run a sweep agent with flag --nowandb")
        wandb.agent(
            sweep_id=args.sid,
            function=lambda: main(is_sweep=True, project_name=None, use_wandb=True),
            count=args.count,
            project=args.projn,
        )
