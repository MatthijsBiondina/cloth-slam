from typing import Optional, Union, Tuple

import numpy as np
from torch import Tensor

from utils.tools import pyout


def fit_wrapped_gaussian(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(logits)
    theta = np.linspace(-np.pi, np.pi, num=logits.size, endpoint=False)
    idx = np.empty((n, n), dtype=int)
    idx[0] = np.arange(n)
    for ii in range(1, n):
        idx[ii] = np.roll(idx[ii - 1], 1)

    theta, logits = theta[idx], logits[idx]
    theta -= 2 * np.pi * np.tril(np.ones_like(theta), -1)

    P = logits / np.sum(logits, axis=-1)[:, None]
    means = np.sum(P * theta, axis=-1)
    scales = np.sum(P * np.abs(theta - means[:, None]), axis=-1)
    argmin = np.argmin(scales)

    mean = (means[argmin] + np.pi) % (2 * np.pi) - np.pi
    scale = scales[argmin]

    return mean, scale


class WrappedGaussian1D:
    def __init__(self, mean: Optional[float] = None,
                 scale: Optional[float] = None,
                 logits: Optional[Union[np.ndarray, Tensor]] = None):
        assert not ((mean is None or scale is None) and logits is None)

        if logits is None:
            self.mean: float = mean
            self.scale: float = scale
        else:
            if isinstance(logits, Tensor):
                logits = logits.detach().cpu().numpy()
            self.mean, self.scale = fit_wrapped_gaussian(logits)

    def heatmap(self, N=100):
        theta = np.linspace(-np.pi, np.pi, num=N, endpoint=False)
        wrapped_distance = np.minimum(
            np.abs(theta - self.mean), 2 * np.pi - np.abs(theta - self.mean))

        logits = np.exp(-.5 * (wrapped_distance / self.scale) ** 2) \
                 / (self.scale * np.sqrt(2 * np.pi))
        logits = logits / np.max(logits)

        return logits


if __name__ == "__main__":
    L = np.array([0.101556525, 0.0918202, 0.08204918, 0.074851096, 0.06900766,
                  0.06351615, 0.05442171, 0.04701652, 0.04015003, 0.030418243,
                  0.024432482, 0.02039022, 0.017155696, 0.0142181,
                  0.011872858, 0.008752666, 0.006726926, 0.00578588,
                  0.005482984, 0.0046639713, 0.0044884356, 0.0039675804,
                  0.004199379, 0.003977055, 0.00470131, 0.004365411,
                  0.00399986, 0.0039814077, 0.003759907, 0.0032647934,
                  0.002825071, 0.0025331972, 0.0024248508, 0.0017727291,
                  0.0016905793, 0.0015036586, 0.0016213676, 0.0018152834,
                  0.0016986706, 0.0016171358, 0.0017512193, 0.001618317,
                  0.002103489, 0.0021001967, 0.0026456083, 0.0036823521,
                  0.006413057, 0.007627073, 0.0095344875, 0.010998965,
                  0.012283031, 0.013108756, 0.016655665, 0.020997334,
                  0.023931598, 0.028240183, 0.03078213, 0.032338828,
                  0.035035294, 0.03857358, 0.037303094, 0.041931547,
                  0.0413505, 0.04793012, 0.051159613, 0.05342264, 0.05415049,
                  0.053166177, 0.05393851, 0.059143122, 0.056165237,
                  0.06182126, 0.07325557, 0.08212302, 0.09307791, 0.116311,
                  0.14017175, 0.16230465, 0.19561857, 0.21928516, 0.24320604,
                  0.25998676, 0.28530344, 0.30160403, 0.30156016, 0.30861264,
                  0.32528952, 0.3188267, 0.3146651, 0.30184698, 0.28668222,
                  0.2711247, 0.24753983, 0.22315338, 0.20602314, 0.17530546,
                  0.16327482, 0.14986001, 0.1334373, 0.116553135])

    fit_wrapped_gaussian(L)
