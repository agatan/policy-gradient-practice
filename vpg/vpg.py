from functools import update_wrapper
from typing import NamedTuple

import numpy as np
import torch
from torch import optim
from torch.utils import tensorboard
import gym
from gym import spaces

from vpg import model


class Experience(NamedTuple):
    obs: torch.Tensor
    act: torch.Tensor
    ret: torch.Tensor


class Buffer:
    def __init__(
        self, obs_dim: int, act_dim: int, size: int, gamma=0.99, lam=0.95
    ) -> None:
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(self, obs: np.ndarray, act: np.ndarray, reward: float,) -> None:
        """Store an experience.

        Args:
            obs (np.ndarray): Observed state.
            act (np.ndarray): Selected action.
            reward (float): Acquired reward.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.reward_buf[self.ptr] = reward
        self.ptr += 1

    def finish_path(self) -> None:
        """Call this method to annotate current path is finished.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.reward_buf[path_slice]
        r = np.sum(rewards)
        self.return_buf[path_slice] = r
        self.path_start_idx = self.ptr

    def get(self) -> Experience:
        assert self.ptr == self.max_size
        return Experience(
            obs=torch.as_tensor(self.obs_buf, dtype=torch.float),
            act=torch.as_tensor(self.act_buf, dtype=torch.float),
            ret=torch.as_tensor(self.return_buf, dtype=torch.float),
        )


def compute_loss(actor: model.Actor, experience: Experience) -> torch.Tensor:
    _, logp = actor(experience.obs, experience.act)
    return -(logp * experience.ret).mean()


def main(env_name: str, epochs: int, steps_per_epoch: int = 500, updates_per_step: int = 10):
    writer = tensorboard.SummaryWriter()

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    assert isinstance(env.action_space, spaces.Discrete)
    act_dim = env.action_space.n

    policy = model.ActorCritic(obs_dim, act_dim)
    actor_optimizer = optim.Adam(policy.actor.parameters())

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        buf = Buffer(obs_dim, act_dim, size=steps_per_epoch)
        obs = env.reset()
        episode_rewards = []
        is_first_episode = True
        for step in range(steps_per_epoch):
            if epoch % 10 == 0 and is_first_episode:
                env.render()
            act = policy.act(torch.as_tensor(obs, dtype=torch.float))
            next_obs, reward, done, _ = env.step(act)
            buf.store(obs, act, reward)
            obs = next_obs
            if is_first_episode:
                episode_rewards.append(reward)
            terminated = step == steps_per_epoch - 1
            if done or terminated:
                buf.finish_path()
            if done:
                is_first_episode = False
                obs = env.reset()
        average_loss = 0
        for _ in range(updates_per_step):
            actor_optimizer.zero_grad()
            loss = compute_loss(policy.actor, buf.get())
            loss.backward()
            actor_optimizer.step()
            average_loss += loss.detach().item() / updates_per_step

        print(f"EpRew: {np.sum(episode_rewards)}, Loss: {average_loss}")
        writer.add_scalar("EpRew", np.sum(episode_rewards))
        writer.add_scalar("Loss", average_loss)



if __name__ == "__main__":
    main("CartPole-v0", 200)
