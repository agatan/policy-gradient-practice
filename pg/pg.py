from typing import NamedTuple
import dataclasses

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
        self, obs: np.ndarray, act: np.ndarray, reward: float, value: float
    ) -> None:
        """Store an experience.

        Args:
            obs (np.ndarray): Observed state.
            act (np.ndarray): Selected action.
            reward (float): Acquired reward.
            value (float): Estimated value.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.reward_buf[self.ptr] = reward
        self.value_buf[self.ptr] = value
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
        )


@dataclasses.dataclass
class TrainerConfig:
    epochs: int
    steps_per_epoch: int
    updates_per_step: int
    render: bool
    use_reward_to_go: bool
    use_actor_critic: bool


class Trainer:
    def __init__(
        self, ac: model.ActorCritic, env: gym.Env, config: TrainerConfig
    ) -> None:
        self.ac = ac
        self.env = env
        self.config = config
        self.actor_optimizer = optim.Adam(self.ac.actor.parameters())
        self.obs_dim = env.observation_space.shape[0]
        self.writer = tensorboard.SummaryWriter()
        self.global_step = 0

    def _train_one_epoch(self, epoch: int):
        print(f"Epoch {epoch}")
        buffer = Buffer(
            self.obs_dim,
            self.config.steps_per_epoch,
            self.config.use_reward_to_go,
            self.config.use_actor_critic,
        )
        obs = self.env.reset()
        episode_rewards = []
        is_first_episode = True
        for step in range(self.config.steps_per_epoch):
            self.global_step += 1
            if self.config.render and epoch % 10 == 0 and is_first_episode:
                self.env.render()
            act, value, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float))
            next_obs, reward, done, _ = self.env.step(act)
            buffer.store(obs, act, reward, value)
            obs = next_obs
            if is_first_episode:
                episode_rewards.append(reward)
            terminated = step == self.config.steps_per_epoch - 1
            if terminated and not done:
                _, value, _ = self.ac.step(torch.as_tensor(obs, dtype=torch.float))
                buffer.finish_path(value)
            if done:
                buffer.finish_path(0)
                is_first_episode = False
                obs = self.env.reset()
        experience = buffer.get()
        average_loss = 0
        for _ in range(self.config.updates_per_step):
            self.actor_optimizer.zero_grad()
            loss = _compute_loss(
                self.ac, experience, self.config.use_actor_critic
            )
            loss.backward()
            self.actor_optimizer.step()
            average_loss += loss.detach().item() / self.config.updates_per_step

        print(f"EpRew: {np.sum(episode_rewards)}, Loss: {average_loss}")
        self.writer.add_scalar(
            "EpRew", np.sum(episode_rewards), global_step=self.global_step
        )
        self.writer.add_scalar("Loss", average_loss, global_step=self.global_step)

    def train(self):
        for epoch in range(1, 1 + self.config.epochs):
            self._train_one_epoch(epoch)


def _compute_loss(
    ac: model.ActorCritic, experience: Experience, use_actor_critic: bool
) -> torch.Tensor:
    _, logp = ac.actor(experience.obs, experience.act)
    a = experience.advantages if use_actor_critic else experience.ret
    loss = -(logp * a).mean()
    if use_actor_critic:
        values = ac.critic(experience.obs)
        critic_loss = F.smooth_l1_loss(values, experience.ret)
        loss += critic_loss
    return loss


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
            render=args.render,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
