import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

# --- Constants for Market Simulation (based on historical data averages) ---
# Format: (Expected_Annual_Return, Standard_Deviation) for a 100% Stock Portfolio
STOCK_METRICS = {
    'low': (0.05, 0.08),     # Conservative blend
    'medium': (0.07, 0.12),  # Balanced blend
    'high': (0.09, 0.15)     # Aggressive blend
}
BOND_RETURN = 0.02
BOND_STD_DEV = 0.03

def get_portfolio_metrics(risk_tolerance, years):
    """
    Calculates the optimized portfolio mix and the resulting blended expected return and volatility.
    Mix is determined by risk and time horizon (de-risking slightly for shorter horizons).
    """
    base_stocks = 0
    if risk_tolerance == 'low':
        base_stocks = 0.45
    elif risk_tolerance == 'medium':
        base_stocks = 0.65
    elif risk_tolerance == 'high':
        base_stocks = 0.85
        
    # Apply a slight time horizon adjustment (more conservative for < 10 years)
    if years < 10:
        stock_ratio = max(0.20, base_stocks * (years / 10))
    else:
        stock_ratio = base_stocks

    bond_ratio = 1.0 - stock_ratio

    # Calculate blended expected return and standard deviation
    stock_return, stock_std = STOCK_METRICS[risk_tolerance]
    
    # Blended Expected Return: R_p = w_s * R_s + w_b * R_b
    expected_annual_return = (stock_ratio * stock_return) + (bond_ratio * BOND_RETURN)
    
    # Blended Volatility (Simplistic model assuming low correlation for visualization)
    # Volatility is weighted average of standard deviation
    annual_std_dev = (stock_ratio * stock_std) + (bond_ratio * BOND_STD_DEV)

    return stock_ratio, bond_ratio, expected_annual_return, annual_std_dev

def calculate_contributions(target_amount, years, current_savings, expected_rate):
    """
    Calculates the required monthly contribution using the Future Value (FV) formula.
    
    Args:
        target_amount (float): The final goal amount.
        years (int): Time horizon in years.
        current_savings (float): Initial amount saved.
        expected_rate (float): Blended annual expected return rate (decimal).
        
    Returns:
        float: Required monthly contribution.
    """
    if years <= 0:
        return target_amount # Immediate goal, just need the target

    monthly_rate = expected_rate / 12
    months = years * 12
    
    # Future Value of Current Savings (FV_CS)
    fv_current_savings = current_savings * (1 + monthly_rate) ** months
    
    # Required Future Value from Contributions (FV_Req_Contrib)
    fv_required_contributions = target_amount - fv_current_savings
    
    if fv_required_contributions <= 0:
        return 0.0 # Goal likely already met

    # Calculate Payment (PMT) required for Annuity (using simplified formula)
    # PMT = FV * [ r / ((1+r)^n - 1) ]
    # Note: Using numpy's fv function for precision might be better, but the formula is clearer.
    monthly_contrib = fv_required_contributions * (monthly_rate / (((1 + monthly_rate) ** months) - 1))
    
    # Adjust for monthly contribution being at the start of the period (BGN mode)
    # This requires a slightly more complex formula or an iterative approach. 
    # For simplicity, we use the END mode formula above and divide by (1 + monthly_rate) for a BGN adjustment.
    # We will stick to the simplified calculation for a general estimate.
    
    return max(0.0, monthly_contrib)

def run_monte_carlo(target_amount, years, monthly_contrib, current_savings, risk_tolerance, num_simulations=500):
    """
    Performs a Monte Carlo simulation for wealth projection.
    
    Args:
        target_amount (float): The final goal amount.
        years (int): Time horizon in years.
        monthly_contrib (float): Monthly contribution amount.
        current_savings (float): Initial amount saved.
        risk_tolerance (str): 'low', 'medium', or 'high'.
        num_simulations (int): Number of paths to simulate.
        
    Returns:
        dict: Simulation results including success probability, median wealth, and full path data.
    """
    months = years * 12
    
    # Get blended portfolio metrics (includes stock/bond mix logic)
    stock_ratio, bond_ratio, expected_annual_return, annual_std_dev = get_portfolio_metrics(risk_tolerance, years)
    
    # Monthly metrics
    monthly_rate_mean = expected_annual_return / 12
    monthly_std_dev = annual_std_dev / np.sqrt(12)
    
    # Initialize results array to store end wealth for all simulations
    final_wealths = np.zeros(num_simulations)
    
    # Initialize array to store the wealth trajectory for each month/year (for charting)
    wealth_paths = np.zeros((num_simulations, months + 1)) 

    for i in range(num_simulations):
        portfolio_value = current_savings
        wealth_paths[i, 0] = portfolio_value

        for month in range(months):
            # Generate a random return from a normal distribution
            monthly_return = np.random.normal(monthly_rate_mean, monthly_std_dev)
            
            # Add growth
            portfolio_value *= (1 + monthly_return)
            
            # Add monthly contribution
            portfolio_value += monthly_contrib
            
            wealth_paths[i, month + 1] = portfolio_value

        final_wealths[i] = portfolio_value

    # Calculate success probability
    success_prob = np.sum(final_wealths >= target_amount) / num_simulations * 100
    
    # Calculate percentiles for charting
    # Transpose the paths array so we can calculate percentiles for each time step (column)
    p05 = np.percentile(wealth_paths, 5, axis=0)
    p50 = np.percentile(wealth_paths, 50, axis=0)
    p95 = np.percentile(wealth_paths, 95, axis=0)

    # Generate the chart
    generate_monte_carlo_chart(years, target_amount, p05, p50, p95, success_prob)

    return {
        'success_prob': float(success_prob),
        'median_wealth': float(p50[-1]),
        'p05_path': p05.tolist(), # Convert to list for BSON compatibility
        'p50_path': p50.tolist(),
        'p95_path': p95.tolist(),
        'investment_mix': predict_optimal_mix(years, risk_tolerance) # Include mix here for consistency
    }

def generate_monte_carlo_chart(years, target_amount, p05, p50, p95, success_prob):
    """
    Generates and saves the Monte Carlo visualization with a modern, minimalist FinTech aesthetic.
    """
    
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Set up a clean, default style
    plt.style.use('default') 
    
    plt.figure(figsize=(10, 6), dpi=120) 
    
    # Create time axis (in years)
    time_points = np.linspace(0, years, len(p50))
    
    # --- Modern Color Palette ---
    COLOR_MEDIAN = '#1e3a8a'      # Deep, professional blue
    COLOR_CONFIDENCE_FILL = '#dbeafe' # Very light blue for background fill
    COLOR_GOAL = '#ef4444'        # Strong red for critical line
    COLOR_BORDER_LINES = '#93c5fd' # Subtler blue for percentile boundaries

    # 1. Plot the Confidence Band (90% interval)
    plt.fill_between(time_points, p05, p95, 
                     color=COLOR_CONFIDENCE_FILL, 
                     alpha=0.6, 
                     label='90% Confidence Band')
    
    # 2. Plot the Median (50th Percentile) - THE FOCUS LINE
    plt.plot(time_points, p50, 
             label='Median Projection', 
             color=COLOR_MEDIAN, 
             linewidth=4, 
             solid_capstyle='round')
             
    # 3. Plot the outer percentile boundaries (subtle)
    plt.plot(time_points, p95, color=COLOR_BORDER_LINES, linestyle='-', linewidth=1, alpha=0.7, label='Optimistic (95th)')
    plt.plot(time_points, p05, color=COLOR_BORDER_LINES, linestyle='-', linewidth=1, alpha=0.7, label='Pessimistic (5th)')
    
    # 4. Plot the target goal line
    plt.axhline(y=target_amount, 
                color=COLOR_GOAL, 
                linestyle='--', 
                linewidth=2, 
                label=f'Goal: ₹{target_amount:,.0f}')
    
    # --- Aesthetics ---
    plt.title(f'Future Value Projection (Success Probability: {success_prob:.2f}%)', 
              fontsize=16, 
              fontweight='bold', 
              color='#1f2937', # Dark gray text
              pad=20)
              
    plt.xlabel('Years', fontsize=12, color='#4b5563')
    plt.ylabel('Portfolio Value (₹)', fontsize=12, color='#4b5563')
    
    # Format Y-axis to display in millions or thousands
    def currency_formatter(x, pos):
        if x >= 1e6:
            return f'₹{x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'₹{x*1e-3:.0f}K'
        return f'₹{x:.0f}'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Remove all spines/borders
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
        
    # Use subtle grid lines
    plt.grid(axis='y', linestyle='-', alpha=0.2, color='#9ca3af')
    plt.tick_params(axis='both', which='major', labelsize=10) 

    # Place legend outside for clarity
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.0), fontsize=9, frameon=False, ncol=2)
    
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(static_dir, 'wealth_chart.png')
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()

def predict_optimal_mix(years, risk_tolerance):
    """
    Determines the optimal stock and bond mix based on risk and time horizon.
    Uses the logic embedded in get_portfolio_metrics to ensure consistency.
    """
    stock_ratio, bond_ratio, _, _ = get_portfolio_metrics(risk_tolerance, years)
    return {
        'stocks': float(f"{stock_ratio:.2f}"), # Ensure floats are rounded for display
        'bonds': float(f"{bond_ratio:.2f}")
    }

# --- Utility function required by app.py ---
def get_expected_rate(risk_tolerance, years=15):
    """Returns the expected return used for the initial contribution calculation."""
    _, _, expected_rate, _ = get_portfolio_metrics(risk_tolerance, years) 
    return expected_rate
