import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from lib.opt_types import Function


class Quadratic:
    def __init__(self, A):
        self.A = A
        self.L = float(np.linalg.eigvalsh(A)[-1])

    def __call__(self, x):
        return 0.5 * np.dot(x, self.A @ x)

    def grad(self, x):
        return self.A @ x


@dataclass
class QuadraticSum(Function):
    components: tuple = ()

    def __getitem__(self, i):
        return self.components[i]


def make_quadratic_problem(seed=0):
    """Construct eleven positive definite quadratics with one large L_i."""
    rng = np.random.default_rng(seed)
    n = d = 11
    constants = rng.integers(11, 112, size=n)
    constants[1] = 1111
    components = []
    for L in constants:
        eigenvalues = rng.random(d)
        eigenvalues -= eigenvalues.min()
        eigenvalues /= eigenvalues.max()
        eigenvalues = 1 + (L - 1) * eigenvalues
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        A = np.dot(Q * eigenvalues, Q.T)
        components.append(Quadratic(0.5 * (A + A.T)))

    mean_A = np.mean([component.A for component in components], axis=0)
    f = QuadraticSum(
        f=lambda x: 0.5 * np.dot(x, mean_A @ x),
        grad=lambda x: mean_A @ x,
        i_grad=lambda i, x: components[i].grad(x),
        minimum=0.0,
        strng_cvx=float(np.linalg.eigvalsh(mean_A)[0]),
        lips_grad=float(np.linalg.eigvalsh(mean_A)[-1]),
        n=n,
        L_max=max(component.L for component in components),
        components=tuple(components),
    )
    return f, np.full(d, 3.0)


def plot_sampling(methods, f, x_zero, max_iteration=250, seed=0):
    """Compare squared distances to the quadratic problem's minimizer, zero."""
    random_state = np.random.get_state()
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        for method in methods:
            np.random.seed(seed)
            state = method.init_state(f, x_zero.copy())
            distances = [np.dot(state.x_k, state.x_k)]
            for _ in range(max_iteration):
                state = method.state_update(f, state)
                distances.append(np.dot(state.x_k, state.x_k))
            ax.semilogy(range(max_iteration + 1), distances, lw=2, label=method.name)
    finally:
        np.random.set_state(random_state)
    ax.set_xlabel("Individual sample gradient evaluations")
    ax.set_ylabel(r"$\|\mathbf{x}^k-\mathbf{x}^\star\|_2^2$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
