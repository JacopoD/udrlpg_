import torch
from udrlpg.utils import scale
import numpy as np
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt
import wandb


class MyEnv:
    def __init__(
        self,
        train_env,
        eval_env,
        train_seed: int,
        eval_seed: int,
        policy,
        device,
        config,
        model,
        tracker,
        rbuf,
    ) -> None:
        self.config = config
        self.model = model
        self.tracker = tracker
        self.rbuf = rbuf

        self.train_ep_count = 0
        self.eval_ep_count = 0
        self.train_env = train_env
        self.eval_env = eval_env
        self.train_seed = train_seed
        self.eval_seed = eval_seed
        self.policy = policy
        self.device = device

        self.is_discrete = isinstance(self.train_env.action_space, Discrete)

        self.global_ep_count = 0
        self.global_steps = 0

        if self.config.use_obs_norm:
            self.n = 0
            self.mean = np.zeros(train_env.observation_space.shape)
            self.mean_diff = np.zeros(train_env.observation_space.shape)
            self.std = np.zeros(train_env.observation_space.shape)

        if self.config.dump_env_obs:
            self.obs_history = []

    def increment_ep_count(self, is_train):
        if is_train:
            self.train_ep_count += 1
        else:
            self.eval_ep_count += 1

        self.global_ep_count += 1

    def do_episode(
        self,
        w: torch.Tensor,
        is_train: bool,
        scale_reward: int = False,
        disable_evaluation: bool = False,
        remove_survival_bonus: bool = False,  # should only be true if is_train is false
    ):
        env = self.train_env if is_train else self.eval_env
        seed = self.train_seed if is_train else self.eval_seed
        ep_count = self.train_ep_count if is_train else self.eval_ep_count

        observation, _ = env.reset(seed=seed + ep_count)

        tot_reward = 0
        done = False

        while not done:
            if self.config.use_obs_norm:
                if is_train:
                    self.push_obs(observation)
                if self.global_steps > 0 or (not self.config.moving_obs_norm):
                    observation = self.normalize_obs(observation)

            obs_tensor = torch.FloatTensor(observation).to(self.device)

            with torch.no_grad():
                action = self.policy.forward(obs_tensor.unsqueeze(0), w).squeeze(0)
                if self.is_discrete:
                    action = action.argmax().item()
                else:
                    action = action.cpu().numpy()

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if ((not self.config.survival_bonus) and is_train) or remove_survival_bonus:
                match self.config.gym_env_name:
                    case "Hopper-v4" | "Ant-v4":
                        tot_reward += reward - 1
                    case _:
                        raise NotImplementedError
            else:
                tot_reward += reward

            if is_train:
                self.global_steps += 1

            if (
                is_train
                and (not disable_evaluation)
                and ((self.global_steps + 1) % self.config.eval_frequency == 0)
            ):
                # evaluate now
                if (self.global_steps + 1) % self.config.extensive_eval_frequency == 0:
                    self.extensive_eval()
                self.eval()

            if is_train and ((self.global_steps + 1) == self.config.max_steps):
                return None

        self.increment_ep_count(is_train)

        if is_train or remove_survival_bonus:
            is_higher = False
            match self.config.gym_env_name:
                case "Hopper-v4" | "Ant-v4":
                    is_higher = (
                        tot_reward > self.config.max_reward - env._max_episode_steps
                    )
                case _:
                    is_higher = tot_reward > self.config.max_reward
            if is_higher:
                print(f"Found higher reward: {tot_reward}")

        # tot_reward = np.clip(tot_reward, self.config.min_reward, self.config.max_reward)
        tot_reward = max(self.config.min_reward, tot_reward)

        if scale_reward:
            return self.scale_from_env_to_my(tot_reward)
        return tot_reward

    def normalize_obs(self, obs: np.ndarray):
        return (obs - self.mean) / (self.std + 1e-8)

    def push_obs(self, obs: np.ndarray):
        if self.config.dump_env_obs:
            self.obs_history.append(obs)
        if not self.config.moving_obs_norm:
            return

        self.n += 1.0
        last_mean = self.mean
        self.mean += (obs - self.mean) / self.n
        self.mean_diff += (obs - last_mean) * (obs - self.mean)
        var = self.mean_diff / (self.n - 1) if self.n > 1 else np.square(self.mean)
        self.std = np.sqrt(var)

    def eval(self):
        self.model.eval()
        n_eval_runs = 5
        rews = []
        weights = []
        with torch.no_grad():
            for _ in range(n_eval_runs):
                if self.config.use_vae:
                    w = self.model.sample(
                        torch.tensor([self.rbuf.max_seen])
                        .unsqueeze(-1)
                        .type(torch.float32)
                        .to(self.device)
                    )
                else:
                    w = self.model.forward(
                        torch.tensor([self.rbuf.max_seen])
                        .unsqueeze(-1)
                        .type(torch.float32)
                        .to(self.device)
                    )
                rews.append(self.do_episode(w.squeeze(0), is_train=False))
                weights.append(w)

        weights = torch.stack(weights)

        # how much do the means of the weights differ when asking for the same command
        weights_mean_std = weights.std(dim=0).mean()
        pairwise_mse = ((weights.unsqueeze(0) - weights.unsqueeze(1)) ** 2).mean()

        self.tracker.weights_std = weights_mean_std
        self.tracker.pairwise_mse = pairwise_mse

        if self.config.buckets:
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.bar(
                x=self.scale_from_my_to_env(self.rbuf.limits),
                height=self.rbuf.get_status(asbool=False),
            )
            self.tracker.rbuf_status_plot = wandb.Image(fig)
            plt.close(fig)

        self.tracker.push_rew(np.mean(rews))

        self.buffer_loss()

        self.tracker.log_to_wandb(env_steps=self.global_steps, extra=None)

        plt.close("all")
        self.model.train()

    def extensive_eval(self):
        self.model.eval()

        # The buffer is filled with policies with reward up to max_reward if survival bonus is active
        # or up to max_reward - max_episode_length (its scaled representation) if survival bonus is off.
        # In both cases the maximum command asked wich when scaled will always be 1 will be the maximum possible
        # value in the replay buffer.
        # Take the example of Hopper, the maximum possible including survival bonus is 3000, if survival bonus is removed
        # the maxmum reward possible is 2000 which will be represented by 1 in the replay buffer.

        n_eval_runs = 5
        commands = np.linspace(self.config.min_r_scale, self.config.max_r_scale, 10)

        rews = np.zeros((*commands.shape, n_eval_runs))

        
        for i, command in enumerate(commands):
            for j in range(n_eval_runs):
                with torch.no_grad():
                    if self.config.use_vae:
                        w = self.model.sample(
                            torch.tensor([command])
                            .unsqueeze(-1)
                            .type(torch.float32)
                            .to(self.device)
                        )
                    else:
                        w = self.model.forward(
                            torch.tensor([command])
                            .unsqueeze(-1)
                            .type(torch.float32)
                            .to(self.device)
                        )
                rews[i, j] = self.do_episode(
                    w.squeeze(0),
                    is_train=False,
                    remove_survival_bonus=(not self.config.survival_bonus),
                )

        scaled_commands = self.scale_from_my_to_env(commands)

        # mean for each reward requested
        means = rews.mean(axis=1)
        std = rews.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(x=scaled_commands, y1=means - std, y2=means + std, alpha=0.25)
        ax.plot(scaled_commands, means)
        ax.plot(scaled_commands, scaled_commands, color="violet")
        ax.scatter(scaled_commands, means, marker=".")
        ax.set_yticks(scaled_commands)
        ax.set_xticks(scaled_commands)
        ax.set_ylabel("actual")
        ax.set_xlabel("requested")

        mse = ((means - scaled_commands) ** 2).mean()

        self.tracker.extensive_eval_plot = wandb.Image(fig)
        self.tracker.extensive_eval_mse = mse

        self.tracker.push_extensive_eval_history(rews)

        self.tracker.log_extensive_eval(self.global_steps)

        plt.close(fig)

        self.model.train()

    def buffer_loss(self):
        if not self.config.buckets:
            return
        rec = np.zeros(self.rbuf.n_buckets)
        if self.config.use_vae:
            reg = np.zeros(self.rbuf.n_buckets)

        for b in self.rbuf.get_nonempty():
            commands, weights = zip(*self.rbuf.pick_n_from(b, 5))
            with torch.no_grad():
                commands = (
                    torch.tensor(commands, dtype=torch.float32)
                    .unsqueeze(-1)
                    .to(self.device)
                )
                weights = torch.vstack(weights).to(self.device)
                if self.config.use_vae:
                    _, decoded, mean, log_var = self.model.forward(weights, commands)
                    rec_loss, reg_loss = self.model.loss(
                        weights, decoded, mean, log_var, return_separately=True
                    )
                    
                    reg[b] = reg_loss.item()
                    rec[b] = rec_loss.item()
                else:
                    decoded = self.model.forward(commands)
                    rec[b] = torch.nn.functional.mse_loss(decoded, weights).item()

                
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(
            x=self.scale_from_my_to_env(self.rbuf.limits),
            height=rec,
        )
        self.tracker.rbuf_rec_loss_plot = wandb.Image(fig)
        plt.close(fig)


        if self.config.use_vae:
            fig2, ax2 = plt.subplots(figsize=(15, 5))
            ax2.bar(
                x=self.scale_from_my_to_env(self.rbuf.limits),
                height=reg,
            )
            self.tracker.rbuf_reg_loss_plot = wandb.Image(fig2)
            plt.close(fig2)


    def scale_from_env_to_my(self, values):
        max_reward = self.config.max_reward

        if not self.config.survival_bonus:
            max_reward -= self.train_env._max_episode_steps

        return scale(
            values,
            self.config.min_reward,
            max_reward,
            self.config.max_r_scale,
            self.config.min_r_scale,
        )

    def scale_from_my_to_env(self, values):
        max_reward = self.config.max_reward
        if not self.config.survival_bonus:
            max_reward -= self.train_env._max_episode_steps

        return scale(
            values,
            self.config.min_r_scale,
            self.config.max_r_scale,
            max_reward,
            self.config.min_reward,
        )
