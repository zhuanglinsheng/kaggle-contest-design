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

    def __init__(self):
        """Initialize the analyzer with parameters from paper appendix."""
        self.contest_params = {}
        self.load_parameters()

    def load_parameters(self) -> None:
        """Load contest parameters from the paper appendix table."""
        print("Loading contest parameters from paper appendix table...")

        for row in CONTEST_PARAMETERS:
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
