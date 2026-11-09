"""Plot the conditional-variance trajectories shown in the manuscript."""
import argparse
from pathlib import Path

import numpy as np


def variance(time, initial, sigma, precision):
    steady = sigma / np.sqrt(precision)
    scaled_time = np.tanh(sigma * np.sqrt(precision) * time)
    return steady * (initial + steady * scaled_time) / (steady + initial * scaled_time)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/runs/figures/figure_S_evolve.pdf'))
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt

    time = np.linspace(0., 10., 1001)
    fig, ax = plt.subplots()
    # The upper initial value is inferred from the retained figure's vector path.
    for initial, style in ((.8, '-'), (1.08330975, '--')):
        ax.plot(time, variance(time, initial, sigma=1., precision=1.), style,
                color='black', linewidth=1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
