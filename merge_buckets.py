import pickle
import os
from udrlpg.dynamic_buckets_fifo import ReplayBufferBucket
from udrlpg.my_new_env import MyEnv
from udrlpg.virtualmlpp import VirtualMLPPolicy
from udrlpg.utils import parse_config, dotdict
import random
import gymnasium as gym
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, default=None)
parser.add_argument("--projn", type=str, required=True, default=None)
# parser.add_argument("--el", type=int, required=False, default=1)
parser.add_argument("--noeval", action="store_true")
parser.add_argument("--usenorm", action="store_true")
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using ", device)

config = {
    "gym_env_name": args.env,
    "survival_bonus": "auto",
    "min_reward": "auto",
    "max_reward": "auto",
    "eval_frequency": "auto",
    "extensive_eval_frequency": "auto",
    "max_steps": "auto",
    "max_r_scale": 1,
    "min_r_scale": 0,
    "use_obs_norm": args.usenorm
}


print(config["gym_env_name"])

config = dotdict(config)
config = parse_config(config, False)

train_gym_env = gym.make(config.gym_env_name)
eval_gym_env = gym.make(config.gym_env_name)

if isinstance(train_gym_env.action_space, gym.spaces.Discrete):
    policy_neurons = (
        [train_gym_env.observation_space.shape[0]]
        + [256, 256]
        + [train_gym_env.action_space.n]
    )
    vmlp = VirtualMLPPolicy(
        layer_sizes=policy_neurons,
        act_lim=1,
        nonlinearity="tanh",
        output_activation="linear",
    )
elif isinstance(train_gym_env.action_space, gym.spaces.Box):
    policy_neurons = (
        [train_gym_env.observation_space.shape[0]]
        + [256, 256]
        + [train_gym_env.action_space.shape[0]]
    )
    vmlp = VirtualMLPPolicy(
        layer_sizes=policy_neurons,
        act_lim=train_gym_env.action_space.high[0],
        nonlinearity="tanh",
        output_activation="tanh",
    )

env = MyEnv(
    train_gym_env,
    eval_gym_env,
    train_seed=100,
    eval_seed=101,
    policy=vmlp,
    device=device,
    config=config,
    tracker=None,
    rbuf=None,
    model=None,
)

if args.usenorm:
    env.train_ep_count = 0
else:
    env.train_ep_count = 1

original_buckets = []
bucket = ReplayBufferBucket(100, random.Random(), 100_000)


obs_means = []
obs_stds = []
obs_histories = []

max_el = -1
for i, d in enumerate(
    os.walk(f"/home/jacopo/storage/models/{args.projn}/{config.gym_env_name}")
):
    if i == 0:
        continue
    
    with open(f"{d[0]}/buckets.pickle", "rb") as fp:
        b = pickle.load(fp)
        for b_idx in b.get_nonempty():
            el = b.pick_from(b_idx)
            max_el = max(el[0], max_el)
            bucket.insert(el, lambda x: x[0])
        original_buckets.append(b)
    if args.usenorm and not args.noeval:
        obs_means.append(np.load(f"{d[0]}/env_obs_mean.npy"))
        obs_stds.append(np.load(f"{d[0]}/env_obs_std.npy"))
        obs_histories.append(np.load(f"{d[0]}/obs_history.npy"))
    print(d[0])

if args.usenorm and not args.noeval:
    assert (
        len(obs_means) == len(obs_stds) == len(original_buckets)
    ), "Error, number of elements must match"

print(f"Found {len(original_buckets)} runs \nBuckets loaded!")

print("----------------------------------------------------------")

if args.usenorm and not args.noeval:
    obs_means = np.vstack(obs_means)
    obs_stds = np.vstack(obs_stds)

if not args.noeval:

    n_runs = 2
    for i, b in enumerate(original_buckets):
        if args.usenorm:
            print(f"Buckets of run {i} with its own observation statistics:")
            print(f"Mean: {obs_means[i]} | Std: {obs_stds[i]}")
            print(
                f"Computed from history: Mean: {obs_histories[i].mean(axis=0)} | Std: {obs_histories[i].std(axis=0)}"
            )
            env.mean = obs_means[i]
            env.std = obs_stds[i]
        else:
            print(f"Buckets of run {i} with no obs norm")

        status = np.full(b.n_buckets, np.nan)
        for idx in b.get_nonempty():
            _, weights = zip(*bucket.buckets[idx])
            status[idx] = 0
            for w in weights:
                status[idx] += np.sum(
                    [
                        env.do_episode(
                            w, is_train=False, scale_reward=False, disable_evaluation=True
                        )
                        for _ in range(n_runs)
                    ]
                )
            status[idx] = status[idx] / (len(weights) * n_runs)
        print(status)

    print("----------------------------------------------------------")

    for i, b in enumerate(original_buckets):
        if args.usenorm:
            print(f"Mean policy of buckets of run {i} with its own observation statistics:")
            env.mean = obs_means[i]
            env.std = obs_stds[i]
        else:
            print(f"Mean policy of buckets of run {i} with no obs norm:")
        status = np.full(b.n_buckets, np.nan)
        for idx in b.get_nonempty():
            _, weights = zip(*bucket.buckets[idx])
            mean_weight = torch.vstack(weights).mean(dim=0)
            status[idx] = np.mean(
                [
                    env.do_episode(
                        w, is_train=False, scale_reward=False, disable_evaluation=True
                    )
                    for _ in range(5)
                ]
            )
        print(status)

    print("----------------------------------------------------------")

    if args.usenorm:

        print(f"Means of observation means: {obs_means.mean(axis=0)}")
        print(f"Means of observation stds (not used): {obs_stds.mean(axis=0)}")

        stacked_obs_hists = np.vstack(obs_histories)

        print(f"Mean obs of merged histories: {stacked_obs_hists.mean(axis=0)}")
        print(f"Std obs of merged histories: {stacked_obs_hists.std(axis=0)}")

        t1 = np.sum(
            [
                (obs_histories[i].shape[0] - 1) * (obs_stds[i] ** 2)
                for i in range(len(obs_histories))
            ],
            axis=0,
        )

        t2 = np.sum(
            [
                (obs_histories[i].shape[0]) * ((obs_means[i] - obs_means.mean(axis=0)) ** 2)
                for i in range(len(obs_histories))
            ],
            axis=0,
        )

        new_merged_std = np.sqrt((t1 + t2) / (len(obs_histories) * obs_histories[0].shape[0]))

        print(f"New std: {new_merged_std}")

    print("----------------------------------------------------------")

if args.usenorm and not args.noeval:
    env.mean = obs_means.mean(axis=0)
    env.std = new_merged_std

print("Info merged buffer:")

print("Max reward in buffer:", max_el)
print(bucket.limits)
print(bucket.total)
print(bucket.get_status(False))

if not args.noeval:
    print("----------------------------------------------------------")

    mean_policies = [None for _ in range(bucket.n_buckets)]
    mean_policies_evals = np.full(bucket.n_buckets, np.nan)

    bucket_eval = np.full(bucket.n_buckets, np.nan)

    for i, b in enumerate(bucket.get_nonempty()):
        print(f"{i}/{bucket.get_nonempty().shape[0]}", end="\r")
        _, weights = zip(*bucket.buckets[b])
        weights = torch.vstack(weights)
        mean_policies[b] = weights.mean(dim=0)
        mean_policies_evals[b] = np.array(
            [
                env.do_episode(mean_policies[b], is_train=False, disable_evaluation=True)
                for _ in range(5)
            ]
        ).mean()

        bucket_eval[b] = 0
        for w in weights:
            bucket_eval[b] += np.array(
                [
                    env.do_episode(w, is_train=False, disable_evaluation=True)
                    for _ in range(2)
                ]
            ).mean()
        bucket_eval[b] = bucket_eval[b] / weights.shape[0]


    print("Mean policy for each bucket:")
    print(mean_policies_evals)
    print("Buckets", bucket_eval)

idd = str(random.randint(0,100_000))
print(idd)

with open(
    f"/home/jacopo/storage/merged_bucket_{args.projn}_{config.gym_env_name}_{idd}.pickle", "wb+"
) as fp:
    pickle.dump(bucket, fp)
