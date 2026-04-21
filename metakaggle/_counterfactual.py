#!/usr/bin/env python3
"""
Counterfactual Analysis for Kaggle Contest Design - Version 2

This script performs counterfactual analysis on prize (theta) parameter
using parameter estimates from the paper appendix table.

Usage:
    python _counterfactual.py [--theta_multipliers 0.5 0.75 1.0 1.25 1.5 1.75 2.0]
                              [--output_dir ./counterfactual_results]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

# Import contest parameters from separate file
from contest_parameters import CONTEST_PARAMETERS

warnings.filterwarnings('ignore')


class CounterfactualAnalyzerV2:
    """Main class for performing counterfactual analysis using paper appendix parameters."""

    # Parameter table from paper appendix (Table 4)
    # Columns: ContestId, N_i, N_j, Prize(k, USD), PublicData%, lambda, sigma, mu_0, c_i, c_j, r, lambda_MLE, sigma_MLE
    PARAMETER_TABLE = [
        # ContestId, N_i, N_j, Prize, PublicData%, lambda, sigma, mu_0, c_i, c_j, r, lambda_MLE, sigma_MLE
        (2445, 7, 19, 5.00, 25, 15.377, 0.980, -0.089, 4.304, 3.154, 2.088, 1539.379, 0.394),
        (2454, 19, 13, 0.15, 62, 15.016, 1.030, 4.377, 0.688, 0.810, 18.035, 932.836, 0.659),
        (2464, 11, 12, 0.95, 20, 9.844, 2.325, -0.474, 1.631, 1.380, 14.217, 99.556, 3.741),
        (2478, 13, 11, 0.95, 30, 13.636, 0.673, -0.444, 3.171, 3.811, 6.333, 19406.675, 0.770),
        (2489, 12, 21, 0.50, 10, 16.650, 0.576, 0.431, 3.535, 2.303, 11.210, 5759.075, 0.843),
        (2549, 13, 15, 0.50, 30, 12.214, 1.357, -1.055, 1.409, 1.157, 14.758, 525.201, 2.891),
        (2551, 48, 30, 1.50, 30, 26.143, 0.539, -0.286, 3.207, 4.141, 6.669, 140015.172, 0.512),
        (2667, 55, 67, 30.00, 50, 16.977, 1.695, 0.758, 3.919, 3.803, 1.184, 1.805, 5.430),
        (2749, 22, 22, 3.00, 42, 20.708, 0.701, 0.906, 4.308, 4.022, 4.367, 37219.066, 0.340),
        (2762, 12, 21, 1.00, 35, 11.659, 0.779, 1.068, 3.103, 2.237, 13.005, 20.923, 1.244),
        (2860, 9, 6, 1.00, 30, 12.735, 0.816, 0.133, 3.096, 3.728, 6.933, 33538.355, 0.034),
        (3064, 13, 5, 0.20, 25, 14.106, 0.630, -0.788, 1.160, 2.780, 13.821, 8206710.757, 0.263),
        (3080, 25, 27, 0.09, 25, 23.442, 0.531, -1.217, 0.623, 0.643, 19.327, 150384.896, 0.187),
        (3288, 20, 10, 1.00, 0, 10.864, 3.729, -1.169, 0.822, 1.750, 15.448, 2.179, 0.616),
        (3338, 31, 29, 1.50, 30, 24.827, 0.541, 0.203, 3.508, 3.345, 7.687, 8200.515, 0.244),
        (3353, 26, 17, 10.00, 30, 16.307, 1.232, -0.249, 3.749, 4.305, 2.066, 22001.196, 1.410),
        (3366, 16, 11, 0.50, 50, 15.653, 0.693, -1.936, 2.055, 3.018, 13.340, 4849.451, 0.499),
        (3507, 9, 13, 0.50, 33, 14.030, 1.055, 0.529, 2.440, 1.891, 14.095, 163.652, 1.101),
        (3509, 8, 13, 0.20, 30, 12.714, 0.741, -0.024, 2.491, 1.513, 13.807, 3167.382, 0.884),
        (3526, 15, 13, 5.00, 20, 15.809, 0.793, -0.053, 4.127, 4.159, 1.280, 11651.555, 0.242),
        (3641, 7, 41, 1.50, 43, 20.406, 0.579, 0.615, 4.292, 1.875, 3.434, 2447.287, 0.319),
        (3774, 29, 26, 0.50, 50, 23.554, 0.537, -0.997, 2.580, 2.898, 13.300, 29.891, 0.952),
        (3800, 21, 28, 3.00, 15, 20.046, 0.820, 3.384, 4.322, 3.599, 6.632, 1174.496, 0.625),
        (3926, 25, 20, 0.30, 25, 22.016, 0.547, -0.047, 1.237, 1.529, 13.393, 4798.921, 0.332),
        (3928, 15, 21, 0.68, 70, 19.525, 0.584, -0.421, 3.604, 3.051, 11.838, 87393.056, 0.139),
        (3929, 35, 31, 8.00, 50, 5.865, 3.653, 0.062, 3.165, 3.520, 10.320, 4794.303, 5.862),
        (3960, 47, 50, 8.00, 40, 28.973, 0.883, -0.992, 4.624, 4.670, 3.583, 624.827, 0.514),
        (4031, 6, 61, 5.00, 30, 19.132, 0.831, 0.605, 4.487, 1.975, 3.021, 125709.402, 1.214),
        (4104, 8, 6, 20.00, 20, 10.864, 2.116, 0.036, 3.729, 3.994, 0.742, 42409.966, 0.141),
        (4366, 31, 31, 8.00, 2, 23.414, 0.913, -0.434, 4.367, 4.554, 2.979, 3601.181, 0.487),
        (4407, 44, 51, 4.00, 19, 14.573, 1.670, 0.070, 3.591, 3.238, 12.165, 56.026, 2.693),
        (4453, 15, 18, 2.00, 30, 17.165, 0.617, 1.204, 4.110, 3.071, 2.607, 47662.222, 0.892),
        (4477, 7, 11, 2.00, 50, 11.870, 1.104, -3.305, 3.867, 3.222, 5.142, 2524.322, 0.325),
        (4488, 11, 8, 2.00, 1, 8.033, 5.347, 1.108, 2.160, 2.843, 12.739, 1903.540, 1.559),
        (4493, 19, 14, 10.00, 40, 16.683, 1.161, -0.191, 3.883, 4.306, 1.567, 7787.118, 0.355),
        (4657, 66, 62, 4.00, 30, 34.581, 0.559, 0.330, 4.657, 4.691, 3.410, 1221277.563, 0.205),
        (4699, 13, 22, 5.00, 30, 18.734, 0.765, 0.112, 4.242, 3.833, 1.207, 223019.351, 0.211),
        (4986, 16, 5, 10.00, 50, 13.378, 1.203, -0.029, 3.287, 4.306, 1.214, 80756.892, 0.393),
        (5056, 18, 35, 5.00, 33, 21.957, 0.720, -0.780, 4.484, 4.094, 2.763, 536274.232, 0.324),
        (5144, 11, 9, 20.00, 20, 7.244, 3.118, 10.283, 3.617, 3.737, 1.411, 3752.980, 2.695),
        (5174, 42, 47, 3.00, 50, 23.823, 0.744, -1.240, 4.346, 4.162, 8.369, 292993.349, 1.622),
        (5229, 28, 31, 15.00, 10, 12.332, 3.417, 0.320, 3.961, 3.564, 6.028, 14.707, 2.880),
        (5261, 45, 43, 10.00, 30, 26.183, 0.965, -1.578, 4.361, 4.651, 2.525, 83664.604, 0.965),
        (5357, 36, 22, 5.00, 30, 22.768, 0.774, 1.093, 4.028, 4.506, 3.022, 8980.088, 0.354),
        (5390, 22, 17, 4.00, 30, 17.724, 0.819, 0.513, 3.650, 4.317, 4.262, 1165.060, 1.086),
        (5497, 44, 24, 4.00, 30, 25.292, 0.652, 0.642, 4.231, 4.571, 3.005, 9784887.950, 0.233),
        (6322, 23, 26, 10.00, 66, 9.106, 2.653, -0.405, 3.753, 3.536, 5.852, 48.308, 3.981),
        (6927, 19, 16, 4.00, 1, 18.477, 0.918, -0.601, 3.925, 4.247, 4.178, 8317.660, 0.272),
        (7115, 19, 39, 4.00, 30, 15.827, 0.848, 0.444, 4.314, 3.226, 3.401, 408806.642, 1.469),
        (7162, 51, 14, 1.00, 50, 20.424, 0.808, -1.269, 1.209, 4.010, 8.932, 39471.599, 1.093),
        (7634, 52, 44, 2.00, 2, 23.259, 0.568, 0.761, 3.767, 4.307, 8.242, 7635.257, 1.509),
        (7878, 5, 7, 4.00, 29, 9.554, 1.484, -0.612, 3.815, 3.285, 1.450, 619.191, 0.875),
        (8076, 34, 29, 6.00, 10, 22.880, 0.806, -1.387, 4.321, 4.419, 2.903, 8518.348, 1.134),
        (8078, 29, 32, 4.00, 36, 8.066, 1.796, 1.664, 3.587, 3.088, 10.469, 143.391, 5.352),
        (8219, 16, 23, 0.40, 22, 15.559, 0.604, -0.087, 2.601, 1.940, 14.337, 674373.206, 1.542),
        (8396, 53, 59, 0.50, 1, 13.606, 1.385, 7.016, 0.880, 0.767, 16.517, 1158.201, 2.451),
        (8540, 30, 38, 5.00, 18, 22.939, 0.840, -0.117, 4.326, 4.212, 4.251, 7531.700, 1.251),
        (9120, 52, 44, 10.00, 20, 28.586, 0.876, -0.006, 4.690, 4.746, 1.789, 49444.336, 0.219),
        (9949, 21, 35, 5.00, 20, 21.022, 0.923, 1.016, 4.539, 3.565, 3.200, 14679.381, 0.563),
        (10200, 22, 25, 4.00, 9, 15.905, 1.513, -7.127, 3.619, 3.583, 6.466, 28779.845, 1.034),
        (10684, 16, 58, 4.00, 57, 18.929, 0.659, -7.058, 4.450, 3.642, 4.785, 32.622, 1.342),
        (13333, 18, 19, 2.00, 25, 19.948, 0.638, 0.168, 3.975, 3.913, 6.246, 85026.315, 0.079),
        (14242, 68, 37, 3.00, 20, 31.135, 0.558, 0.383, 4.266, 4.663, 5.096, 2915.546, 0.372),
        (14420, 56, 20, 20.00, 22, 13.393, 1.571, -3.492, 3.144, 4.609, 1.464, 5.884, 5.899),
        (18045, 40, 50, 4.00, 30, 26.932, 0.654, -3.178, 4.269, 4.344, 3.532, 187.042, 0.809),
        (19018, 63, 67, 8.00, 30, 29.425, 0.835, 1.891, 4.634, 4.360, 2.918, 18544.514, 0.863),
        (19991, 22, 50, 4.00, 20, 24.460, 0.658, 0.062, 4.599, 3.990, 3.472, 925.409, 1.058),
        (20270, 20, 16, 4.00, 30, 18.848, 0.815, -0.255, 3.707, 4.250, 3.938, 293.230, 0.531),
        (21669, 38, 37, 1.00, 21, 17.621, 0.831, -1.649, 1.917, 2.520, 14.959, 2382.409, 2.604),
        (22962, 23, 18, 4.00, 24, 17.168, 0.879, -1.295, 3.760, 4.158, 4.583, 7906.999, 1.143),
        (23249, 21, 40, 1.00, 16, 24.123, 0.532, 0.538, 2.978, 2.075, 7.226, 1795533.078, 0.101),
        (23652, 24, 24, 1.00, 20, 20.607, 0.643, -0.193, 2.862, 3.456, 14.124, 87142.434, 1.056),
        (37077, 11, 46, 4.00, 24, 19.329, 0.886, 0.584, 4.311, 2.254, 3.507, 10417.255, 1.563),
        (38128, 27, 52, 8.00, 42, 25.657, 0.808, -0.562, 4.687, 4.444, 1.655, 23943.751, 0.263),
        (38760, 10, 17, 5.00, 33, 12.412, 1.155, 0.026, 4.192, 3.255, 2.936, 6387644.371, 2.551),
    ]

    def __init__(self):
        """Initialize the analyzer with parameters from paper appendix."""
        self.contest_params = {}
        self.load_parameters()

    def load_parameters(self) -> None:
        """Load contest parameters from the paper appendix table."""
        print("Loading contest parameters from paper appendix table...")

        for row in self.PARAMETER_TABLE:
            contest_id = str(row[0])
            self.contest_params[contest_id] = {
                'contest_id': contest_id,
                'N_i': row[1],
                'N_j': row[2],
                'theta': row[3],  # Prize in k USD
                'public_data_percent': row[4],
                'lambda': row[5],
                'sigma': row[6],
                'mu_0': row[7],
                'c_i': row[8],
                'c_j': row[9],
                'r': row[10],
                'lambda_MLE': row[11],
                'sigma_MLE': row[12]
            }

        print(f"Loaded parameters for {len(self.contest_params)} contests")

    @staticmethod
    def fn_gamma(u: float) -> float:
        """Gamma function from Ryvkin's model."""
        if (u < -1) or (u > 1):
            raise ValueError(f"fn_gamma(x): -1 < x < 1; found x = {u}")
        if u == -1:
            return -np.inf
        elif u == 1:
            return np.inf
        else:
            return u / (1 - u**2) + np.arctanh(u)

    @staticmethod
    def fn_invgamma(x: float) -> float:
        """Inverse gamma function."""
        return np.arctan(0.856 * x) * 2 / np.pi

    @staticmethod
    def fn_rho(z: float, gamma_rho_i: float, gamma_rho_j: float) -> float:
        """Rho function."""
        cdf_value = norm.cdf(z, 0, 1)
        # Ensure scalar value
        if hasattr(cdf_value, '__len__'):
            cdf_value = float(cdf_value)
        loc = cdf_value * (gamma_rho_i + gamma_rho_j) - gamma_rho_j
        return float(CounterfactualAnalyzerV2.fn_invgamma(float(loc)))

    @staticmethod
    def compute_equilibrium_efforts(
        y: float,
        t: float,
        T: float,
        theta: float,
        sigma: float,
        c_i: float,
        c_j: float
    ) -> Tuple[float, float]:
        """Compute equilibrium effort levels for both players."""
        if T <= t:
            raise ValueError(f"t must be less than T; found t={t}, T={T}")

        sigma_power_2 = sigma**2
        w_i = theta / (sigma_power_2 * c_i)
        w_j = theta / (sigma_power_2 * c_j)

        # Compute rho values
        rho_i = (np.exp(w_i) + np.exp(-w_j) - 2) / (np.exp(w_i) - np.exp(-w_j))
        rho_j = (np.exp(w_j) + np.exp(-w_i) - 2) / (np.exp(w_j) - np.exp(-w_i))

        gamma_rho_i = CounterfactualAnalyzerV2.fn_gamma(rho_i)
        gamma_rho_j = CounterfactualAnalyzerV2.fn_gamma(rho_j)

        # Compute effort levels
        y_stderr = sigma * np.sqrt(T - t)
        z = y / y_stderr if y_stderr > 0 else 0
        rho_z = CounterfactualAnalyzerV2.fn_rho(z, gamma_rho_i, gamma_rho_j)

        K = sigma_power_2 / 2 * (gamma_rho_i + gamma_rho_j) * (1 - rho_z**2)

        # Normal density
        density_y = norm.pdf(y, 0, y_stderr)

        m_i = float(density_y * K * (1 + rho_z))
        m_j = float(density_y * K * (1 - rho_z))

        return m_i, m_j

    def simulate_contest_dynamics(
        self,
        contest_id: str,
        theta_multiplier: float = 1.0,
        N_Delta: int = 100,
        Delta2f: float = 0.01
    ) -> Dict:
        """
        Simulate contest dynamics with modified theta (prize).

        Args:
            contest_id: Contest identifier
            theta_multiplier: Multiplier for original theta
            N_Delta: Number of time steps
            Delta2f: Time step size

        Returns:
            Dictionary with equilibrium effort paths and summary statistics
        """
        if contest_id not in self.contest_params:
            raise ValueError(f"Contest {contest_id} not found in parameters")

        contest_params = self.contest_params[contest_id]

        # Use modified theta
        original_theta = contest_params['theta']
        theta = original_theta * theta_multiplier

        # Extract parameters
        mu_0 = contest_params['mu_0']
        c_i = contest_params['c_i']
        c_j = contest_params['c_j']
        sigma = contest_params['sigma']
        lambda_param = contest_params['lambda']

        # Initialize arrays
        m_i_path = np.zeros(N_Delta)
        m_j_path = np.zeros(N_Delta)
        tilde_y = np.zeros(N_Delta + 1)
        tilde_y[0] = mu_0

        T = N_Delta * Delta2f

        # Simulate dynamics
        for i in range(N_Delta):
            t = i * Delta2f
            m_i, m_j = self.compute_equilibrium_efforts(
                y=tilde_y[i],
                t=t,
                T=T,
                theta=theta,
                sigma=sigma,
                c_i=c_i,
                c_j=c_j
            )

            m_i_path[i] = m_i
            m_j_path[i] = m_j

            # Update filtered state (simplified)
            # In counterfactual analysis, we don't have observed hat_y for modified theta
            kalman_gain = np.sqrt(lambda_param) * sigma * 0  # Simplified
            tilde_y[i + 1] = tilde_y[i] + (m_i - m_j + kalman_gain) * Delta2f

        # Calculate summary statistics
        total_effort_i = np.sum(m_i_path) * Delta2f
        total_effort_j = np.sum(m_j_path) * Delta2f
        total_effort = total_effort_i + total_effort_j
        max_effort_i = np.max(m_i_path)
        max_effort_j = np.max(m_j_path)
        max_effort = max(max_effort_i, max_effort_j)

        return {
            'contest_id': contest_id,
            'm_i_path': m_i_path,
            'm_j_path': m_j_path,
            'tilde_y': tilde_y,
            'total_effort_i': total_effort_i,
            'total_effort_j': total_effort_j,
            'total_effort': total_effort,
            'max_effort_i': max_effort_i,
            'max_effort_j': max_effort_j,
            'max_effort': max_effort,
            'theta_used': theta,
            'theta_multiplier': theta_multiplier,
            'original_theta': original_theta
        }

    def run_counterfactual_analysis(
        self,
        theta_multipliers: Optional[List[float]] = None,
        N_Delta: int = 100,
        Delta2f: float = 0.01
    ) -> pd.DataFrame:
        """
        Run counterfactual analysis for all contests.

        Args:
            theta_multipliers: List of theta multipliers to test
            N_Delta: Number of time steps
            Delta2f: Time step size

        Returns:
            DataFrame with counterfactual simulation results
        """
        if theta_multipliers is None:
            theta_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        all_results = []

        print(f"Running counterfactual analysis for {len(self.contest_params)} contests...")
        print(f"Theta multipliers: {theta_multipliers}")
        print(f"Time steps: N_Delta={N_Delta}, Delta2f={Delta2f}")

        for contest_id in self.contest_params:
            print(f"  Processing contest {contest_id}...")

            # Run counterfactual analysis for each theta multiplier
            for theta_mult in theta_multipliers:
                try:
                    results = self.simulate_contest_dynamics(
                        contest_id=contest_id,
                        theta_multiplier=theta_mult,
                        N_Delta=N_Delta,
                        Delta2f=Delta2f
                    )

                    # Store results
                    all_results.append({
                        'contest_id': contest_id,
                        'theta_multiplier': theta_mult,
                        'original_theta': results['original_theta'],
                        'modified_theta': results['theta_used'],
                        'total_effort': results['total_effort'],
                        'max_effort': results['max_effort'],
                        'total_effort_i': results['total_effort_i'],
                        'total_effort_j': results['total_effort_j'],
                        'max_effort_i': results['max_effort_i'],
                        'max_effort_j': results['max_effort_j'],
                        'N_i': self.contest_params[contest_id]['N_i'],
                        'N_j': self.contest_params[contest_id]['N_j'],
                        'sigma': self.contest_params[contest_id]['sigma'],
                        'lambda': self.contest_params[contest_id]['lambda'],
                        'c_i': self.contest_params[contest_id]['c_i'],
                        'c_j': self.contest_params[contest_id]['c_j']
                    })
                except Exception as e:
                    print(f"    Error for theta_mult={theta_mult}: {e}")
                    continue

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        print(f"Generated {len(results_df)} counterfactual simulations")

        return results_df

    def calculate_percentage_changes(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentage changes relative to baseline (theta_multiplier = 1.0).

        Args:
            results_df: DataFrame with counterfactual simulation results

        Returns:
            DataFrame with percentage changes
        """
        print("Calculating percentage changes relative to baseline...")

        # Create a copy to avoid modifying the original
        df = results_df.copy()

        # For each contest, get baseline values (theta_multiplier = 1.0)
        baseline_data = []

        for contest_id in df['contest_id'].unique():
            contest_df = df[df['contest_id'] == contest_id]

            # Find baseline (theta_multiplier = 1.0)
            baseline = contest_df[contest_df['theta_multiplier'] == 1.0]

            if len(baseline) > 0:
                baseline_row = baseline.iloc[0]
                baseline_data.append({
                    'contest_id': contest_id,
                    'baseline_total_effort': baseline_row['total_effort'],
                    'baseline_max_effort': baseline_row['max_effort'],
                    'baseline_total_effort_i': baseline_row['total_effort_i'],
                    'baseline_total_effort_j': baseline_row['total_effort_j'],
                    'baseline_max_effort_i': baseline_row['max_effort_i'],
                    'baseline_max_effort_j': baseline_row['max_effort_j']
                })

        baseline_df = pd.DataFrame(baseline_data)

        # Merge baseline data with results
        df = pd.merge(df, baseline_df, on=['contest_id'], how='left')

        # Calculate percentage changes
        df['pct_change_total_effort'] = 100 * (df['total_effort'] - df['baseline_total_effort']) / df['baseline_total_effort']
        df['pct_change_max_effort'] = 100 * (df['max_effort'] - df['baseline_max_effort']) / df['baseline_max_effort']
        df['pct_change_total_effort_i'] = 100 * (df['total_effort_i'] - df['baseline_total_effort_i']) / df['baseline_total_effort_i']
        df['pct_change_total_effort_j'] = 100 * (df['total_effort_j'] - df['baseline_total_effort_j']) / df['baseline_total_effort_j']
        df['pct_change_max_effort_i'] = 100 * (df['max_effort_i'] - df['baseline_max_effort_i']) / df['baseline_max_effort_i']
        df['pct_change_max_effort_j'] = 100 * (df['max_effort_j'] - df['baseline_max_effort_j']) / df['baseline_max_effort_j']

        # Calculate theta percentage change
        df['pct_change_theta'] = 100 * (df['theta_multiplier'] - 1.0)

        return df

    def aggregate_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results by theta multiplier.

        Args:
            results_df: DataFrame with percentage changes

        Returns:
            Aggregated DataFrame with mean and standard deviation
        """
        print("Aggregating results by theta multiplier...")

        # Group by theta multiplier
        agg_results = results_df.groupby('theta_multiplier').agg({
            'pct_change_total_effort': ['mean', 'std', 'count'],
            'pct_change_max_effort': ['mean', 'std', 'count'],
            'pct_change_total_effort_i': ['mean', 'std'],
            'pct_change_total_effort_j': ['mean', 'std'],
            'pct_change_max_effort_i': ['mean', 'std'],
            'pct_change_max_effort_j': ['mean', 'std'],
            'pct_change_theta': 'first'
        }).round(2)

        # Flatten column names
        agg_results.columns = ['_'.join(col).strip() for col in agg_results.columns.values]

        return agg_results

    def plot_results(
        self,
        results_df: pd.DataFrame,
        output_dir: str = "./counterfactual_results"
    ) -> None:
        """
        Create visualization plots for counterfactual analysis results.

        Args:
            results_df: DataFrame with percentage changes
            output_dir: Directory to save plots
        """
        print(f"Creating visualization plots in {output_dir}...")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set plot style
        plt.rcParams.update({
            'font.size': 12,
            'figure.figsize': (10, 6),
            'figure.autolayout': True
        })

        # 1. Plot: Percentage change in total effort vs theta change
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total effort
        ax = axes[0, 0]
        for contest_id in results_df['contest_id'].unique()[:20]:  # Plot first 20 contests for clarity
            contest_df = results_df[results_df['contest_id'] == contest_id]
            ax.plot(contest_df['pct_change_theta'], contest_df['pct_change_total_effort'],
                   'o-', alpha=0.5, markersize=3, linewidth=0.5)

        ax.set_xlabel('Percentage Change in Prize (θ)')
        ax.set_ylabel('Percentage Change in Total Effort')
        ax.set_title('Total Effort Response to Prize Changes')
        ax.grid(True, alpha=0.3)

        # Max effort
        ax = axes[0, 1]
        for contest_id in results_df['contest_id'].unique()[:20]:
            contest_df = results_df[results_df['contest_id'] == contest_id]
            ax.plot(contest_df['pct_change_theta'], contest_df['pct_change_max_effort'],
                   'o-', alpha=0.5, markersize=3, linewidth=0.5)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Percentage Change in Prize (θ)')
        ax.set_ylabel('Percentage Change in Max Effort')
        ax.set_title('Max Effort Response to Prize Changes')
        ax.grid(True, alpha=0.3)

        # Player i total effort
        ax = axes[1, 0]
        for contest_id in results_df['contest_id'].unique()[:20]:
            contest_df = results_df[results_df['contest_id'] == contest_id]
            ax.plot(contest_df['pct_change_theta'], contest_df['pct_change_total_effort_i'],
                   'o-', alpha=0.5, markersize=3, linewidth=0.5, color='blue', label='Player i')

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Percentage Change in Prize (θ)')
        ax.set_ylabel('Percentage Change in Total Effort (Player i)')
        ax.set_title('Player i Total Effort Response')
        ax.grid(True, alpha=0.3)

        # Player j total effort
        ax = axes[1, 1]
        for contest_id in results_df['contest_id'].unique()[:20]:
            contest_df = results_df[results_df['contest_id'] == contest_id]
            ax.plot(contest_df['pct_change_theta'], contest_df['pct_change_total_effort_j'],
                   'o-', alpha=0.5, markersize=3, linewidth=0.5, color='red', label='Player j')

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Percentage Change in Prize (θ)')
        ax.set_ylabel('Percentage Change in Total Effort (Player j)')
        ax.set_title('Player j Total Effort Response')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'individual_contest_responses.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Plot: Average response across all contests
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Calculate average response
        avg_response = results_df.groupby('pct_change_theta').agg({
            'pct_change_total_effort': 'mean',
            'pct_change_max_effort': 'mean',
            'pct_change_total_effort_i': 'mean',
            'pct_change_total_effort_j': 'mean'
        }).reset_index()

        # Total effort average
        ax = axes[0]
        ax.plot(avg_response['pct_change_theta'], avg_response['pct_change_total_effort'],
               'o-', linewidth=2, markersize=8, color='darkblue')

        # Calculate standard deviation for fill_between
        std_total = results_df.groupby('pct_change_theta')['pct_change_total_effort'].std().reset_index()
        std_total = std_total.rename(columns={'pct_change_total_effort': 'std'})
        avg_response_with_std = pd.merge(avg_response, std_total, on='pct_change_theta', how='left')

        ax.fill_between(avg_response_with_std['pct_change_theta'],
                       avg_response_with_std['pct_change_total_effort'] - avg_response_with_std['std'],
                       avg_response_with_std['pct_change_total_effort'] + avg_response_with_std['std'],
                       alpha=0.2, color='blue')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Percentage Change in Prize (θ)')
        ax.set_ylabel('Percentage Change in Total Effort')
        ax.set_title('Average Total Effort Response')
        ax.grid(True, alpha=0.3)

        # Max effort average
        ax = axes[1]
        ax.plot(avg_response['pct_change_theta'], avg_response['pct_change_max_effort'],
               'o-', linewidth=2, markersize=8, color='darkgreen')

        # Calculate standard deviation for fill_between
        std_max = results_df.groupby('pct_change_theta')['pct_change_max_effort'].std().reset_index()
        std_max = std_max.rename(columns={'pct_change_max_effort': 'std'})
        avg_response_with_std_max = pd.merge(avg_response, std_max, on='pct_change_theta', how='left')

        ax.fill_between(avg_response_with_std_max['pct_change_theta'],
                       avg_response_with_std_max['pct_change_max_effort'] - avg_response_with_std_max['std'],
                       avg_response_with_std_max['pct_change_max_effort'] + avg_response_with_std_max['std'],
                       alpha=0.2, color='green')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Percentage Change in Prize (θ)')
        ax.set_ylabel('Percentage Change in Max Effort')
        ax.set_title('Average Max Effort Response')
        ax.grid(True, alpha=0.3)

        # Player comparison
        ax = axes[2]
        ax.plot(avg_response['pct_change_theta'], avg_response['pct_change_total_effort_i'],
               'o-', linewidth=2, markersize=8, color='blue', label='Player i')
        ax.plot(avg_response['pct_change_theta'], avg_response['pct_change_total_effort_j'],
               'o-', linewidth=2, markersize=8, color='red', label='Player j')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Percentage Change in Prize (θ)')
        ax.set_ylabel('Percentage Change in Total Effort')
        ax.set_title('Average Response by Player')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_responses.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Plot: Elasticity distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Calculate elasticity for each contest (slope of response)
        elasticities = []
        for contest_id in results_df['contest_id'].unique():
            contest_df = results_df[results_df['contest_id'] == contest_id]
            if len(contest_df) > 1:
                # Simple linear regression for elasticity
                x = contest_df['pct_change_theta'].values
                y = contest_df['pct_change_total_effort'].values
                if len(x) > 1 and not np.all(x == x[0]):
                    slope = np.polyfit(x, y, 1)[0]
                    elasticities.append({
                        'contest_id': contest_id,
                        'elasticity': slope,
                        'original_theta': contest_df['original_theta'].iloc[0]
                    })

        if elasticities:
            elasticity_df = pd.DataFrame(elasticities)

            # Histogram of elasticities
            ax = axes[0]
            ax.hist(elasticity_df['elasticity'], bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(x=elasticity_df['elasticity'].mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {elasticity_df["elasticity"].mean():.3f}')
            ax.set_xlabel('Elasticity (ΔEffort% / ΔPrize%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Effort Elasticities')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Elasticity vs original theta
            ax = axes[1]
            ax.scatter(elasticity_df['original_theta'], elasticity_df['elasticity'],
                      alpha=0.6, edgecolors='black')
            ax.set_xlabel('Original Prize (θ)')
            ax.set_ylabel('Elasticity')
            ax.set_title('Elasticity vs Original Prize Level')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'elasticity_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Save elasticity data
            elasticity_df.to_csv(os.path.join(output_dir, 'elasticity_results.csv'), index=False)

        print(f"Plots saved to {output_dir}")


def main():
    """Main function to run counterfactual analysis."""
    parser = argparse.ArgumentParser(description='Run counterfactual analysis for prize (theta) parameter')
    parser.add_argument('--theta_multipliers', type=float, nargs='+',
                       default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                       help='Theta multipliers to test')
    parser.add_argument('--N_Delta', type=int, default=100,
                       help='Number of time steps')
    parser.add_argument('--Delta2f', type=float, default=0.01,
                       help='Time step size')
    parser.add_argument('--output_dir', type=str, default='./counterfactual_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CounterfactualAnalyzerV2()

    # Run counterfactual analysis
    results_df = analyzer.run_counterfactual_analysis(
        theta_multipliers=args.theta_multipliers,
        N_Delta=args.N_Delta,
        Delta2f=args.Delta2f
    )

    if len(results_df) == 0:
        print("No results generated. Exiting.")
        return

    # Calculate percentage changes
    results_with_pct = analyzer.calculate_percentage_changes(results_df)

    # Aggregate results
    aggregated_results = analyzer.aggregate_results(results_with_pct)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_with_pct.to_csv(os.path.join(args.output_dir, 'counterfactual_results.csv'), index=False)
    aggregated_results.to_csv(os.path.join(args.output_dir, 'aggregated_results.csv'))

    print("\n=== Aggregated Results ===")
    print(aggregated_results)

    # Create visualizations
    analyzer.plot_results(results_with_pct, output_dir=args.output_dir)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total simulations: {len(results_with_pct)}")
    print(f"Number of contests: {results_with_pct['contest_id'].nunique()}")
    print(f"Average elasticity (total effort): {results_with_pct.groupby('pct_change_theta')['pct_change_total_effort'].mean().mean():.3f}")
    print(f"Average elasticity (max effort): {results_with_pct.groupby('pct_change_theta')['pct_change_max_effort'].mean().mean():.3f}")

    # Save summary report
    with open(os.path.join(args.output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Counterfactual Analysis Summary Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total simulations: {len(results_with_pct)}\n")
        f.write(f"Number of contests: {results_with_pct['contest_id'].nunique()}\n")
        f.write(f"Theta multipliers tested: {args.theta_multipliers}\n")
        f.write(f"Time steps: N_Delta={args.N_Delta}, Delta2f={args.Delta2f}\n\n")

        f.write("Aggregated Results:\n")
        f.write(aggregated_results.to_string())
        f.write("\n\n")

        f.write("Key Findings:\n")
        f.write("1. Average total effort elasticity: ")
        f.write(f"{results_with_pct.groupby('pct_change_theta')['pct_change_total_effort'].mean().mean():.3f}\n")
        f.write("2. Average max effort elasticity: ")
        f.write(f"{results_with_pct.groupby('pct_change_theta')['pct_change_max_effort'].mean().mean():.3f}\n")
        f.write("3. Player i average elasticity: ")
        f.write(f"{results_with_pct.groupby('pct_change_theta')['pct_change_total_effort_i'].mean().mean():.3f}\n")
        f.write("4. Player j average elasticity: ")
        f.write(f"{results_with_pct.groupby('pct_change_theta')['pct_change_total_effort_j'].mean().mean():.3f}\n")

    print(f"\nResults saved to {args.output_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
