import numpy as np


class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1

    def update(self, name, parameter, gradient):

        if name not in self.m:
            self.m[name] = np.zeros_like(parameter)
            self.v[name] = np.zeros_like(parameter)

        self.m[name] = (
            self.beta1 * self.m[name]
            + (1 - self.beta1) * gradient
        )

        self.v[name] = (
            self.beta2 * self.v[name]
            + (1 - self.beta2) * gradient ** 2
        )

        m_hat = self.m[name] / (
            1 - self.beta1 ** self.t
        )

        v_hat = self.v[name] / (
            1 - self.beta2 ** self.t
        )

        parameter -= (
            self.lr
            * m_hat
            / (np.sqrt(v_hat) + self.epsilon)
        )