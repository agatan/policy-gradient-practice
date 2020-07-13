from typing import NamedTuple
import dataclasses

import numpy as np
import torch
from torch import optim
from torch.utils import tensorboard
import gym
from gym import spaces
import scipy.signal

from pg import model


class Experience(NamedTuple):
    obs: torch.Tensor
    act: torch.Tensor
    ret: torch.Tensor


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
    def __init__(self, obs_dim: int, size: int, use_reward_to_go: bool, gamma=0.99, lam=0.95) -> None:
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.use_reward_to_go = use_reward_to_go
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
        if self.use_reward_to_go:
            self.return_buf[path_slice] = _discount_cumsum(rewards, self.gamma)
        else:
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


class Trainer:
    @dataclasses.dataclass
    class Config:
        epochs: int
        steps_per_epoch: int
        updates_per_step: int
        render: bool
        use_reward_to_go: bool

    def __init__(self, ac: model.ActorCritic, env: gym.Env, config: Config) -> None:
        self.ac = ac
        self.env = env
        self.config = config
        self.actor_optimizer = optim.Adam(self.ac.actor.parameters())
        self.obs_dim = env.observation_space.shape[0]
        self.writer = tensorboard.SummaryWriter()
        self.global_step = 0

    def _train_one_epoch(self, epoch: int):
        print(f"Epoch {epoch}")
        buffer = Buffer(self.obs_dim, self.config.steps_per_epoch, self.config.use_reward_to_go)
        obs = self.env.reset()
        episode_rewards = []
        is_first_episode = True
        for step in range(self.config.steps_per_epoch):
            self.global_step += 1
            if self.config.render and epoch % 10 == 0 and is_first_episode:
                self.env.render()
            act = self.ac.act(torch.as_tensor(obs, dtype=torch.float))
            next_obs, reward, done, _ = self.env.step(act)
            buffer.store(obs, act, reward)
            obs = next_obs
            if is_first_episode:
                episode_rewards.append(reward)
            terminated = step == self.config.steps_per_epoch - 1
            if done or terminated:
                buffer.finish_path()
            if done:
                is_first_episode = False
                obs = self.env.reset()
        experience = buffer.get()
        average_loss = 0
        for _ in range(self.config.updates_per_step):
            self.actor_optimizer.zero_grad()
            loss = _compute_loss(self.ac.actor, experience)
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


def _compute_loss(actor: model.Actor, experience: Experience) -> torch.Tensor:
    _, logp = actor(experience.obs, experience.act)
    return -(logp * experience.ret).mean()


def main(
    env_name: str,
    epochs: int,
    steps_per_epoch: int = 500,
    updates_per_step: int = 10,
    render: bool = False,
    use_reward_to_go: bool = True,
):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    assert isinstance(env.action_space, spaces.Discrete)
    ac = model.ActorCritic(obs_dim, env.action_space.n)
    trainer = Trainer(
        ac,
        env,
        Trainer.Config(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            updates_per_step=updates_per_step,
            render=render,
            use_reward_to_go=use_reward_to_go,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main("CartPole-v0", 200)
