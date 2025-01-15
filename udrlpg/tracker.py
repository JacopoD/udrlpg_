import os
import datetime
import json
import torch
import wandb
import numpy as np

class Tracker:
    def __init__(
        self, save_path, folder_name, config, dump_config=False, config_dict=None
    ) -> None:
        self.path = self.create_directoy(save_path, folder_name)
        if dump_config:
            if config_dict is None:
                raise BaseException("dump_config is True but no config_dict was passed")
            self.dump_config(config_dict)

        self.config = config

        self.eval_rewards = []
        self.eval_last_rewards = []
        self.position = 0

        self.loss = []
        self.reg_loss = []
        self.rec_loss = []

        self.decoder_log_sigma = []

        self.max_reward_seen = -1

        self.rbuf_status_plot = None
        self.rbuf_rec_loss_plot = None
        self.rbuf_reg_loss_plot = None

        self.weights_std = None
        self.pairwise_mse = None

        self.extensive_eval_history = []
        self.extensive_eval_plot = None
        self.extensive_eval_mse = None

        self.rollout_history_after_noise = []
        self.rollout_history_before_noise = []

    def log_to_wandb(self, env_steps, extra=None):
        to_log = {
            "eval_reward": self.eval_rewards[-1],
            "eval_average_rewards": np.mean(self.eval_rewards),
            "eval_average_last_rewards": np.mean(self.eval_last_rewards),
            "loss": np.mean(self.loss),
            "rbuf_max_reward_seen": self.max_reward_seen,
            "weights_std": self.weights_std,
            "pairwise_mse": self.pairwise_mse,
        }

        if self.config.use_vae:
            to_log["reg_loss"] = np.mean(self.reg_loss)
            to_log["rec_loss"] = np.mean(self.rec_loss)

        if self.config.learn_sigma:
            to_log["decoder_log_sigma"] = np.mean(self.decoder_log_sigma)

        if self.config.buckets:
            to_log["rbuf_status"] = self.rbuf_status_plot
            to_log["rbuf_rec_loss"] = self.rbuf_rec_loss_plot
            to_log["rbuf_reg_loss"] = self.rbuf_reg_loss_plot

        if extra is not None and isinstance(extra, dict):
            to_log.update(extra)

        wandb.log(to_log, step=env_steps)
        self.loss = []
        self.reg_loss = []
        self.rec_loss = []

        self.decoder_log_sigma = []

    def log_extensive_eval(self, env_steps):
        wandb.log(
            {
                "extensive_eval": self.extensive_eval_plot,
                "extensive_eval_mse": self.extensive_eval_mse,
            },
            step=env_steps,
        )

    def push_rew(self, rew):
        if len(self.eval_last_rewards) < 20:
            self.eval_last_rewards.append(rew)
        else:
            self.eval_last_rewards[self.position] = rew
            self.position = (self.position + 1) % 20
        self.eval_rewards.append(rew)

    def push_extensive_eval_history(self, rew_data):
        self.extensive_eval_history.append(rew_data)

    def push_rollout_history(self, before_noise, after_noise):
        self.rollout_history_before_noise.append(before_noise)
        self.rollout_history_after_noise.append(after_noise)

    def dump_config(self, config_dict):
        with open(f"{self.path}/config.json", "w") as f:
            json.dump(config_dict, f)

    def create_directoy(self, path, name):
        new_path = f'{path}/{name.lower()}_{str(datetime.datetime.now()).replace(" ","_").replace(":",".")}'
        os.makedirs(new_path)
        return new_path

    def save_model(self, state_dict, epochs):
        if epochs is None:
            torch.save(state_dict, f"{self.path}/model_best.pt")
        else:
            torch.save(state_dict, f"{self.path}/model_{str(epochs)}.pt")

    def _save_buckets_to_file(self, rbuf):
        if self.config.dump_buckets:
            with open(f"{self.path}/buckets.pickle", "wb") as fp:
                rbuf.dump_buckets(fp)

    def _save_obs_history_to_file(self, env):
        if self.config.use_obs_norm:
            np.save(f"{self.path}/env_obs_mean.npy", env.mean)
            np.save(f"{self.path}/env_obs_std.npy", env.std)
            np.save(f"{self.path}/env_obs_mean_diff.npy", env.mean_diff)
        if self.config.dump_env_obs:
            np.save(f"{self.path}/obs_history.npy", np.array(env.obs_history))

    def done(self):
        np.save(
            f"{self.path}/extensive_eval_data.npy",
            np.array(self.extensive_eval_history),
        )
        wandb.finish()
        
