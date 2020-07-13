from typing import NamedTuple, Optional, Tuple
import dataclasses
from concurrent import futures

import numpy as np
import torch
from torch import optim
from torch.utils import tensorboard
import torch.nn.functional as F
import gym
from gym import spaces
import scipy.signal

from pg import model


class Experience(NamedTuple):
    obs: torch.Tensor
    act: torch.Tensor
    ret: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    logp: torch.Tensor


def _discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    def __init__(
        self,
        obs_dim: int,
        size: int,
        use_reward_to_go: bool,
        use_actor_critic: bool,
        gamma=0.99,
        lam=0.95,
    ) -> None:
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.value_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.use_reward_to_go = use_reward_to_go
        self.use_actor_critic = use_actor_critic
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(
        self, obs: np.ndarray, act: np.ndarray, reward: float, value: float, logp: float
    ) -> None:
        """Store an experience.

        Args:
            obs (np.ndarray): Observed state.
            act (np.ndarray): Selected action.
            reward (float): Acquired reward.
            value (float): Estimated value.
            logp (float): log probability of the selected action in the current policy.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.reward_buf[self.ptr] = reward
        self.value_buf[self.ptr] = value
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_value: float) -> None:
        """Call this method to annotate current path is finished.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.reward_buf[path_slice]
        if self.use_reward_to_go:
            self.return_buf[path_slice] = _discount_cumsum(rewards, self.gamma)
        else:
            r = np.sum(rewards)
            self.return_buf[path_slice] = r
        rews = np.append(self.reward_buf[path_slice], last_value)
        vals = np.append(self.value_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = _discount_cumsum(deltas, self.gamma * self.lam)
        self.path_start_idx = self.ptr

    def get(self) -> Experience:
        assert self.ptr == self.max_size
        self.ptr = 0
        self.path_start_idx = 0
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        return Experience(
            obs=torch.as_tensor(self.obs_buf, dtype=torch.float),
            act=torch.as_tensor(self.act_buf, dtype=torch.float),
            ret=torch.as_tensor(self.return_buf, dtype=torch.float),
            advantages=torch.as_tensor(self.adv_buf, dtype=torch.float),
            values=torch.as_tensor(self.value_buf, dtype=torch.float),
            logp=torch.as_tensor(self.logp_buf, dtype=torch.float),
        )


@dataclasses.dataclass
class TrainerConfig:
    epochs: int
    steps_per_epoch: int
    updates_per_step: int
    render: bool
    use_reward_to_go: bool
    use_actor_critic: bool
    use_ppo: bool


class Actor:
    def __init__(self, env: gym.Env, ac: model.ActorCritic) -> None:
        self.env = env
        self.ac = ac
        self.obs_dim = env.observation_space.shape[0]

    def get_experience(
        self,
        steps_per_epoch: int,
        use_reward_to_go: bool,
        use_actor_critic: bool,
        render: bool,
    ) -> Tuple[float, Experience]:
        """Get an experience for required steps.

        Args:
            steps_per_epoch (int): [description]

        Returns:
            float: the reward of the first episode.
            Experience: the corrected experience.
        """
        buffer = Buffer(
            self.obs_dim, steps_per_epoch, use_reward_to_go, use_actor_critic
        )
        is_first_episode = True
        episode_rewards = []
        obs = self.env.reset()
        for step in range(steps_per_epoch):
            if render and is_first_episode:
                self.env.render()
            act, value, logp = self.ac.step(torch.as_tensor(obs, dtype=torch.float))
            next_obs, reward, done, _ = self.env.step(act)
            buffer.store(obs, act, reward, value, logp)
            obs = next_obs
            if is_first_episode:
                episode_rewards.append(reward)
            terminated = step == steps_per_epoch - 1
            if terminated and not done:
                _, value, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float))
                buffer.finish_path(value)
            if done:
                buffer.finish_path(0)
                is_first_episode = False
                obs = self.env.reset()
        experience = buffer.get()
        reward = np.sum(episode_rewards)
        return reward, experience


class Trainer:
    def __init__(
        self, ac: model.ActorCritic, env: gym.Env, config: TrainerConfig
    ) -> None:
        self.ac = ac
        self.actor = Actor(env, ac)
        self.config = config
        self.actor_optimizer = optim.Adam(self.ac.actor.parameters())
        self.critic_optimizer = optim.Adam(self.ac.critic.parameters())
        self.obs_dim = env.observation_space.shape[0]
        self.writer = tensorboard.SummaryWriter()
        self.global_step = 0

    def _train_one_epoch(self, epoch: int):
        print(f"Epoch {epoch}")
        episode_reward, experience = self.actor.get_experience(
            self.config.steps_per_epoch,
            self.config.use_reward_to_go,
            self.config.use_actor_critic,
            self.config.render and epoch % 10 == 0,
        )
        average_actor_loss = 0
        average_critic_loss = 0
        for _ in range(self.config.updates_per_step):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss, critic_loss = _compute_loss(
                self.ac, experience, self.config.use_actor_critic, self.config.use_ppo
            )
            actor_loss.backward()
            self.actor_optimizer.step()
            average_actor_loss += (
                actor_loss.detach().item() / self.config.updates_per_step
            )
            if critic_loss is not None:
                critic_loss.backward()
                self.critic_optimizer.step()
                average_critic_loss += (
                    critic_loss.detach().item() / self.config.updates_per_step
                )

        print(
            f"EpRew: {episode_reward}, Actor Loss: {average_actor_loss}, Critic Loss: {average_critic_loss}"
        )
        self.writer.add_scalar("EpRew", episode_reward, global_step=epoch)
        self.writer.add_scalar("Actor Loss", average_actor_loss, global_step=epoch)
        self.writer.add_scalar(
            "Critic Loss", average_critic_loss, global_step=self.global_step
        )

    def train(self):
        for epoch in range(1, 1 + self.config.epochs):
            self._train_one_epoch(epoch)


def _compute_loss(
    ac: model.ActorCritic, experience: Experience, use_actor_critic: bool, use_ppo: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _, logp = ac.actor(experience.obs, experience.act)
    if use_actor_critic and use_ppo:
        ratio = torch.exp(logp - experience.logp)
        a = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * experience.advantages
        actor_loss = -(torch.min(ratio * experience.advantages, a)).mean()
    else:
        actor_loss = -(logp * experience.ret).mean()
    critic_loss = None
    if use_actor_critic:
        values = ac.critic(experience.obs)
        critic_loss = F.mse_loss(values, experience.ret)
    return actor_loss, critic_loss


@dataclasses.dataclass
class Config:
    trainer: TrainerConfig
    env_name: str = "CartPole-v0"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--updates_per_step", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--disable_reward_to_go", action="store_true")
    parser.add_argument("--disable_actor_critic", action="store_true")
    parser.add_argument("--disable_ppo", action="store_true")
    args = parser.parse_args()
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    assert isinstance(env.action_space, spaces.Discrete)
    ac = model.ActorCritic(obs_dim, env.action_space.n)
    trainer = Trainer(
        ac,
        env,
        TrainerConfig(
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            updates_per_step=args.updates_per_step,
            use_reward_to_go=not args.disable_reward_to_go,
            use_actor_critic=not args.disable_actor_critic,
            use_ppo=not args.disable_ppo,
            render=args.render,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
