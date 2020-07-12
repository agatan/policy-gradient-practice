import numpy as np
import scipy.signal
import torch
from torch import optim
import torch.utils.tensorboard
import gym

from ppo import actor_critic


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


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, size: int, gamma=0.99, lam=0.95
    ) -> None:
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(
        self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float
    ) -> None:
        """Append a timestep of agent-environment interaction.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = _discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = _discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self) -> None:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def main(
    env_name,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=1000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
):
    writer = torch.utils.tensorboard.SummaryWriter()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape or 1

    ac = actor_critic.MLPActorCritic(obs_dim, env.action_space)
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # A function to compute policy loss
    def compute_loss_pi(obs, act, adv, logp_old):
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # A function to compute value loss
    def compute_loss_v(obs, ret):
        return ((ac.v(obs) - ret) ** 2).mean()

    pi_optimizer = optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = optim.Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, _ = compute_loss_pi(
            data["obs"], data["act"], data["adv"], data["logp"]
        )
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data["obs"], data["ret"]).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(
                data["obs"], data["act"], data["adv"], data["logp"]
            )
            kl = np.mean(pi_info["kl"])
            if kl > 1.5 * target_kl:
                print("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            pi_optimizer.step()

        # Train value function
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data["obs"], data["ret"])
            loss_v.backward()
            vf_optimizer.step()

        for k, v in dict(LossPi=pi_l_old, LossV=v_l_old).items():
            writer.add_scalar(k, v)

    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float))
            if epoch % 20 == 0:
                env.render()
            next_o, r, done, _ = env.step(a)
            ep_len += 1
            ep_ret += r

            # Save the experience
            buf.store(o, a, r, v, logp)

            # Update observation
            o = next_o

            # Terminal?
            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1
            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # 中断のばあいは、次の step の estimated value を最終 value として使う
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    writer.add_scalar("EpRet", ep_ret)
                    writer.add_scalar("EpLen", ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        update()


if __name__ == "__main__":
    main("CartPole-v1", steps_per_epoch=200, epochs=800)
