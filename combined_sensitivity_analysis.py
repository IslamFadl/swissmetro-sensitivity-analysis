#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_sensitivity_analysis.py
================================

This script demonstrates how to estimate a multinomial logit model for the
Swissmetro dataset without relying on the ``biogeme`` package and then
perform a sensitivity analysis.  The model replicates the structure of the
example in the original python file sensitivity_analysis.py by estimating generic time and cost coefficients
shared across all alternatives and two alternative specific constants (the
Swissmetro constant is normalised to zero for identification).  After
estimating the parameters, the script computes how market shares change when
travel times or costs for each mode are increased or decreased.  The
results are visualised in a 2×3 grid of plots.

Usage:
    python3 combined_sensitivity_analysis.py

The script expects ``swissmetro.dat`` to be present in the current working
directory.  It writes the chart to ``sensitivity_analysis_chart.png``.
"""

import numpy as np
import pandas as pd
import matplotlib

# Use a non‑interactive backend so the script can run in headless mode
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Read the Swissmetro data and apply the same filtering as in the
    provided example script.  Only trips with PURPOSE 1 or 3 are kept and
    observations without a valid choice (CHOICE == 0) are discarded.

    Parameters
    ----------
    path : str
        Path to the ``swissmetro.dat`` file.

    Returns
    -------
    pandas.DataFrame
        The filtered dataset with scaled time and cost variables.
    """
    df = pd.read_csv(path, sep='\t')
    # Apply the same filtering as in the original script
    mask = ((df['PURPOSE'] == 1) | (df['PURPOSE'] == 3)) & (df['CHOICE'] != 0)
    df = df.loc[mask].copy()

    # Define cost variables that are zero for GA travel card holders
    ga = df['GA']
    df['TRAIN_COST'] = df['TRAIN_CO'] * (ga == 0)
    df['SM_COST'] = df['SM_CO'] * (ga == 0)
    # Scale time and cost variables by 100 as in the example script
    df['TRAIN_TT_SCALED'] = df['TRAIN_TT'] / 100
    df['SM_TT_SCALED'] = df['SM_TT'] / 100
    df['CAR_TT_SCALED'] = df['CAR_TT'] / 100
    df['TRAIN_COST_SCALED'] = df['TRAIN_COST'] / 100
    df['SM_COST_SCALED'] = df['SM_COST'] / 100
    df['CAR_COST_SCALED'] = df['CAR_CO'] / 100
    return df


def estimate_logit_parameters(df: pd.DataFrame) -> np.ndarray:
    """Estimate the alternative specific constants and generic coefficients
    for travel time and cost by maximizing the multinomial log‑likelihood.

    The model uses Swissmetro as the base alternative (its constant is
    normalised to zero) and imposes generic parameters for time and cost
    across all alternatives.  Availability constraints are respected by
    excluding unavailable alternatives from the choice set on a per
    observation basis.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered data containing scaled variables and availability flags.

    Returns
    -------
    numpy.ndarray
        Estimated parameters in the order [ASC_CAR, ASC_TRAIN, B_TIME, B_COST].
    """
    # Extract arrays for quicker computations
    tt_train = df['TRAIN_TT_SCALED'].to_numpy()
    tt_sm = df['SM_TT_SCALED'].to_numpy()
    tt_car = df['CAR_TT_SCALED'].to_numpy()

    cost_train = df['TRAIN_COST_SCALED'].to_numpy()
    cost_sm = df['SM_COST_SCALED'].to_numpy()
    cost_car = df['CAR_COST_SCALED'].to_numpy()

    # Availability: a mode is available if its availability indicator is non‑zero
    av_train = (df['TRAIN_AV'] * (df['SP'] != 0)).to_numpy()
    av_sm = df['SM_AV'].to_numpy()
    av_car = (df['CAR_AV'] * (df['SP'] != 0)).to_numpy()

    choice = df['CHOICE'].to_numpy()

    def neg_log_likelihood(params: np.ndarray) -> float:
        """Negative log‑likelihood of the multinomial logit model.

        Parameters
        ----------
        params : numpy.ndarray
            Parameter vector [ASC_CAR, ASC_TRAIN, B_TIME, B_COST].

        Returns
        -------
        float
            Negative log‑likelihood.
        """
        asc_car, asc_train, b_time, b_cost = params
        # Utility functions
        v1 = asc_train + b_time * tt_train + b_cost * cost_train
        v2 = b_time * tt_sm + b_cost * cost_sm  # ASC_SM = 0
        v3 = asc_car + b_time * tt_car + b_cost * cost_car

        # Apply availability: if unavailable, contribution is zero
        e1 = np.exp(v1) * (av_train != 0)
        e2 = np.exp(v2) * (av_sm != 0)
        e3 = np.exp(v3) * (av_car != 0)
        denom = e1 + e2 + e3

        # Compute choice probabilities
        p1 = e1 / denom
        p2 = e2 / denom
        p3 = e3 / denom

        # Probability of chosen alternative
        probs = np.where(choice == 1, p1, np.where(choice == 2, p2, p3))
        # Avoid log(0) by clipping probabilities
        return -np.sum(np.log(np.clip(probs, 1e-300, None)))

    # Initial parameter guess
    init_guess = np.array([-0.1, -0.5, -1.0, -1.0])
    # Use Nelder–Mead for robustness; BFGS can sometimes struggle with
    # flat regions in this likelihood
    result = minimize(neg_log_likelihood, init_guess, method='Nelder-Mead')
    return result.x


def perform_sensitivity_analysis(params: np.ndarray, df: pd.DataFrame) -> None:
    """Carry out the sensitivity analysis and generate a chart showing
    how modal probabilities change when travel times or costs for each
    alternative are varied.

    Parameters
    ----------
    params : numpy.ndarray
        Estimated parameter vector [ASC_CAR, ASC_TRAIN, B_TIME, B_COST].
    df : pandas.DataFrame
        Filtered dataset containing scaled variables.

    Notes
    -----
    This function computes baseline averages of travel time and cost and then
    varies either the time or the cost of one alternative at a time from 50 % of
    its baseline value up to 150 %.  All other attributes are held constant at
    their baseline values.  The resulting choice probabilities are plotted in
    a 2×3 grid: the first row shows sensitivity to travel time, and the
    second row shows sensitivity to cost.  Each column corresponds to a
    different alternative being varied.
    """
    asc_car, asc_train, b_time, b_cost = params

    # Compute baseline mean times and costs
    base_tt_train = df['TRAIN_TT_SCALED'].mean()
    base_tt_sm = df['SM_TT_SCALED'].mean()
    base_tt_car = df['CAR_TT_SCALED'].mean()

    base_cost_train = df['TRAIN_COST_SCALED'].mean()
    base_cost_sm = df['SM_COST_SCALED'].mean()
    base_cost_car = df['CAR_COST_SCALED'].mean()

    def compute_probs(t_train, t_sm, t_car, c_train, c_sm, c_car):
        v1 = asc_train + b_time * t_train + b_cost * c_train
        v2 = b_time * t_sm + b_cost * c_sm  # ASC_SM = 0
        v3 = asc_car + b_time * t_car + b_cost * c_car
        denom = np.exp(v1) + np.exp(v2) + np.exp(v3)
        return np.exp(v1) / denom, np.exp(v2) / denom, np.exp(v3) / denom

    # Range of multipliers from 0.5 to 1.5
    factors = np.linspace(0.5, 1.5, 21)

    # Containers for results
    results = {}
    for alt in ['train', 'sm', 'car']:
        # Vary travel time for one alternative at a time
        res_time = {'train': [], 'sm': [], 'car': []}
        for f in factors:
            tt_train = base_tt_train * (f if alt == 'train' else 1)
            tt_sm = base_tt_sm * (f if alt == 'sm' else 1)
            tt_car = base_tt_car * (f if alt == 'car' else 1)
            p_train, p_sm, p_car = compute_probs(
                tt_train, tt_sm, tt_car,
                base_cost_train, base_cost_sm, base_cost_car,
            )
            res_time['train'].append(p_train)
            res_time['sm'].append(p_sm)
            res_time['car'].append(p_car)
        results[f'time_{alt}'] = res_time

        # Vary cost for one alternative at a time
        res_cost = {'train': [], 'sm': [], 'car': []}
        for f in factors:
            c_train = base_cost_train * (f if alt == 'train' else 1)
            c_sm = base_cost_sm * (f if alt == 'sm' else 1)
            c_car = base_cost_car * (f if alt == 'car' else 1)
            p_train, p_sm, p_car = compute_probs(
                base_tt_train, base_tt_sm, base_tt_car,
                c_train, c_sm, c_car,
            )
            res_cost['train'].append(p_train)
            res_cost['sm'].append(p_sm)
            res_cost['car'].append(p_car)
        results[f'cost_{alt}'] = res_cost

    # Create the plot: 2 rows (time, cost) × 3 columns (train, SM, car)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    alts = ['train', 'sm', 'car']
    colors = {'train': 'tab:blue', 'sm': 'tab:orange', 'car': 'tab:green'}
    for i, alt in enumerate(alts):
        # Time sensitivity plots
        ax = axes[0, i]
        res = results[f'time_{alt}']
        for mode in ['train', 'sm', 'car']:
            ax.plot(
                factors,
                res[mode],
                label=mode.capitalize() if i == 0 else None,
                color=colors[mode],
            )
        ax.set_title(f'Time sensitivity: vary {alt}')
        ax.set_xlabel('Time multiplier')
        ax.set_ylabel('Choice probability')
        ax.grid(True, linestyle='--', alpha=0.5)
        # Only show legend in the first subplot of the row
        if i == 0:
            ax.legend()

        # Cost sensitivity plots
        ax2 = axes[1, i]
        res = results[f'cost_{alt}']
        for mode in ['train', 'sm', 'car']:
            ax2.plot(
                factors,
                res[mode],
                label=mode.capitalize() if i == 0 else None,
                color=colors[mode],
            )
        ax2.set_title(f'Cost sensitivity: vary {alt}')
        ax2.set_xlabel('Cost multiplier')
        ax2.set_ylabel('Choice probability')
        ax2.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax2.legend()

    fig.tight_layout()
    fig.savefig('combined_sensitivity_analysis_chart.png', dpi=200)


def main() -> None:
    # Load data
    df = load_and_prepare_data('swissmetro.dat')
    # Estimate model parameters
    params = estimate_logit_parameters(df)
    # Perform sensitivity analysis and generate chart
    perform_sensitivity_analysis(params, df)
    print('Sensitivity analysis completed. Chart saved to combined_sensitivity_analysis_chart.png')


if __name__ == '__main__':
    main()