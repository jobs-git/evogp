import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".01"

import jax, jax.numpy as jnp
from brax import envs
import torch
from typing import Callable
from evogp.tree import Forest


def to_jax_array(x: torch.Tensor) -> jax.Array:
    return jax.dlpack.from_dlpack(x.detach())


def from_jax_array(x: jax.Array) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(x)


class BraxProblem:
    def __init__(
        self,
        env_name: str,
        max_episode_length: int,
        seed: int = 42,
        pop_size: int | None = None,
        backend: str | None = None,
        output_transform: Callable = torch.tanh,
    ):
        self.env: envs.Env = (
            envs.get_environment(env_name=env_name)
            if backend is None
            else envs.get_environment(env_name=env_name, backend=backend)
        )

        self.batch_reset = jax.jit(jax.vmap(self.env.reset))
        self.batch_setp = jax.jit(jax.vmap(self.env.step))

        self.max_episode_length = max_episode_length

        self.pop_size = pop_size
        self.output_transform = output_transform
        self.randkey = jax.random.PRNGKey(seed)

    def evaluate(self, forest: Forest):
        if self.pop_size is None:
            self.pop_size = len(forest)

        self.randkey, subkey = jax.random.split(self.randkey)
        reset_keys = jax.random.split(subkey, self.pop_size)
        brax_states = self.batch_reset(reset_keys)

        done = jnp.zeros(self.pop_size, dtype=bool)
        total_reward = jnp.zeros(self.pop_size)

        for _ in range(self.max_episode_length):
            # jax observations to pytorch
            observations = from_jax_array(brax_states.obs)

            # torch action to jax
            torch_output = forest.forward(observations)
            torch_action = self.output_transform(torch_output)
            actions = to_jax_array(torch_action)

            # jax step
            brax_states = self.batch_setp(brax_states, actions)

            observations, reward, currunt_done = (
                brax_states.obs,
                brax_states.reward,
                brax_states.done,
            )

            # update reward
            total_reward += reward * ~done

            # update done
            done = jnp.logical_or(done, currunt_done)

            # break if done
            if jnp.all(done):
                break

        return from_jax_array(total_reward)

    @property
    def obs_dim(self):
        return self.env.observation_size

    @property
    def action_dim(self):
        return self.env.action_size