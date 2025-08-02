import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def generate_sensitivity_chart():
    """
    This function calculates and visualizes the sensitivity of transport mode
    choice to changes in the cost of the Swissmetro service.
    """
    # --- Part 1: Model Coefficients & Baseline Calculation ---
    # These are the pre-estimated parameters from the biogeme model from the sesnsitivity_analysis.py and swissmetro.dat files.
    B_TIME = -1.12
    B_COST = -1.08
    ASC_TRAIN = -0.669
    ASC_CAR = -0.732
    ASC_SM = 0.0

    # Load and prepare the dataset just like in the original script
    try:
        df = pd.read_csv('swissmetro.dat', sep='\t')
    except FileNotFoundError: # Handle the case where the file is not found
        # If the file is not found, print an error message and exit
        print("Error: 'swissmetro.dat' not found. Make sure the file is in the same directory.")
        return

    remove = (((df.PURPOSE != 1) & (df.PURPOSE != 3)) | (df.CHOICE == 0))
    df.drop(df[remove].index, inplace=True)

    df['SM_COST'] = df['SM_CO'] * (df['GA'] == 0)
    df['TRAIN_COST'] = df['TRAIN_CO'] * (df['GA'] == 0)
    df['TRAIN_TT_SCALED'] = df['TRAIN_TT'] / 100
    df['TRAIN_COST_SCALED'] = df['TRAIN_COST'] / 100
    df['SM_TT_SCALED'] = df['SM_TT'] / 100
    df['SM_COST_SCALED'] = df['SM_COST'] / 100
    df['CAR_TT_SCALED'] = df['CAR_TT'] / 100
    df['CAR_CO_SCALED'] = df['CAR_CO'] / 100

    # Calculate the average 'typical' trip to use as a baseline
    baseline = {
        'train_time': df['TRAIN_TT_SCALED'].mean(),
        'train_cost': df['TRAIN_COST_SCALED'].mean(),
        'sm_time': df['SM_TT_SCALED'].mean(),
        'sm_cost': df['SM_COST_SCALED'].mean(),
        'car_time': df['CAR_TT_SCALED'].mean(),
        'car_cost': df['CAR_CO_SCALED'].mean()
    }

    # --- Part 2: Sensitivity Calculation ---
    def calculate_market_shares(sm_cost_var):
        """Calculates market shares for a given SM cost."""
        v_train = ASC_TRAIN + (B_TIME * baseline['train_time']) + (B_COST * baseline['train_cost'])
        v_sm = ASC_SM + (B_TIME * baseline['sm_time']) + (B_COST * sm_cost_var)
        v_car = ASC_CAR + (B_TIME * baseline['car_time']) + (B_COST * baseline['car_cost'])
        
        exp_v_sm = np.exp(v_sm)
        exp_v_train = np.exp(v_train)
        exp_v_car = np.exp(v_car)
        
        sum_exp_v = exp_v_sm + exp_v_train + exp_v_car
        
        return {
            'Swissmetro': exp_v_sm / sum_exp_v * 100,
            'Train': exp_v_train / sum_exp_v * 100,
            'Car': exp_v_car / sum_exp_v * 100
        }

    # Create a range of cost changes to analyze
    changes = np.linspace(-0.5, 0.5, 50) # Use more points for a smoother curve
    cost_values = baseline['sm_cost'] * (1 + changes)
    
    # Calculate shares for each cost value
    results = [calculate_market_shares(cost) for cost in cost_values]
    results_df = pd.DataFrame(results)
    results_df['change'] = changes * 100

    # --- Part 3: Create a clear Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot the data with clear styles
    ax.plot(results_df['change'], results_df['Swissmetro'], label='Swissmetro', color='#0077B6', linewidth=3.5, zorder=3)
    ax.plot(results_df['change'], results_df['Train'], label='Train', color='#6c757d', linestyle='--', linewidth=2)
    ax.plot(results_df['change'], results_df['Car'], label='Car', color='#d9534f', linestyle='--', linewidth=2)

    # Add titles and labels
    ax.set_title('Customer Sensitivity to Changes in Swissmetro Cost', fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Percentage Change in Swissmetro Cost', fontsize=12)
    ax.set_ylabel('Resulting Market Share', fontsize=12)
    
    # Format axes for clarity
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add a legend
    legend = ax.legend(title='Mode of Transport', fontsize=11, title_fontsize=12)
    legend.get_frame().set_alpha(0.9)

    # Highlight the baseline (0% change) point
    baseline_share = results_df.loc[results_df['change'].abs().idxmin()]
    ax.plot(0, baseline_share['Swissmetro'], 'o', color='#003b5c', markersize=8, zorder=4)
    ax.text(0, baseline_share['Swissmetro'] + 2, f"Baseline: {baseline_share['Swissmetro']:.1f}%", 
            ha='center', fontsize=11, fontweight='bold')
    
    # Add a key insight annotation
    ax.annotate(
        'A 25% price increase causes\na significant drop in ridership',
        xy=(25, results_df.loc[results_df['change'] > 24.9, 'Swissmetro'].iloc[0]),
        xytext=(10, 35),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        bbox=dict(boxstyle="round,pad=0.5", fc="lemonchiffon", ec="black", lw=1),
        fontsize=11, ha='center'
    )

    # Final styling
    fig.tight_layout()
    
    # Save the figure
    plt.savefig('sensitivity_analysis_chart.png', dpi=300)
    print("Chart saved as 'sensitivity_analysis_chart.png'")

if __name__ == '__main__':
    generate_sensitivity_chart()
