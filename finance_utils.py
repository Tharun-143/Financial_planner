import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# --- Configuration ---
RETURNS = {'stocks': 0.07, 'bonds': 0.03}
VOLATILITY = {'stocks': 0.15, 'bonds': 0.05}

# --- Allocation Helper ---
def get_allocation(risk):
    """Maps risk tolerance to a simple stock/bond allocation."""
    if risk == 'low': return {'stocks': 0.2, 'bonds': 0.8}
    elif risk == 'medium': return {'stocks': 0.5, 'bonds': 0.5}
    else: return {'stocks': 0.8, 'bonds': 0.2}

# --- 1. Monthly Contribution Calculation ---
def calculate_contributions(target, years, current, rate=0.05):
    """Calculates the required monthly contribution. Returns a standard Python float."""
    months = years * 12
    # Ensure rate is always treated as a number
    if not isinstance(rate, (int, float)):
        rate = 0.05
        
    monthly_rate = rate / 12
    
    if monthly_rate == 0:
        return float(max((target - current) / months, 0)) if months > 0 else float(max(target - current, 0))
        
    future_value_of_current = current * (1 + monthly_rate)**months
    future_value_of_contributions_factor = (((1 + monthly_rate)**months - 1) / monthly_rate)
    
    if future_value_of_contributions_factor <= 0:
        return float(max(target - current, 0))
        
    contrib = (target - future_value_of_current) / future_value_of_contributions_factor
    
    # Ensures return is a standard Python float
    return float(max(contrib, 0))

# --- 2. Monte Carlo Simulation ---
def run_monte_carlo(target, years, monthly_contrib, current_savings, risk):
    """
    Runs Monte Carlo simulation. 
    Returns dictionary with success_prob and median_wealth as standard Python floats.
    """
    sims = 1000
    months = years * 12
    alloc = get_allocation(risk)
    
    portfolio_return = alloc['stocks'] * RETURNS['stocks'] + alloc['bonds'] * RETURNS['bonds']
    portfolio_vol = alloc['stocks'] * VOLATILITY['stocks'] + alloc['bonds'] * VOLATILITY['bonds']
    
    endings = []
    for _ in range(sims):
        wealth = current_savings # Start with the current_savings parameter
        for m in range(months):
            # Monthly return is drawn from a normal distribution
            r = np.random.normal(portfolio_return / 12, portfolio_vol / np.sqrt(12))
            wealth = wealth * (1 + r) + monthly_contrib
        endings.append(wealth)
    
    endings_array = np.array(endings)
    success_prob = np.mean(endings_array >= target) * 100
    median_wealth = np.median(endings_array)
    
    # FIX: Chart generation is now explicitly saved to the 'static' folder
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(endings_array, bins=50, density=True, color='#2c5282', alpha=0.7)
        plt.axvline(target, color='r', linestyle='dashed', linewidth=2, label='Target')
        plt.title('Monte Carlo Simulation of Final Wealth')
        plt.xlabel('Final Wealth ($)')
        plt.ylabel('Probability Density')
        
        # Ensure 'static' folder exists in the root directory for Flask
        if not os.path.exists('static'):
            os.makedirs('static')
            
        plt.savefig('static/wealth_chart.png') 
        plt.close()
    except Exception as e:
        # In a real app, logging this failure is important
        print(f"Error saving chart: {e}") 
        pass
    
    # Convert all NumPy results to standard Python floats for BSON compatibility
    return {
        'success_prob': float(success_prob), 
        'median_wealth': float(median_wealth)
    }

# --- 3. Optimal Mix Prediction (ML) ---
def predict_optimal_mix(years, risk):
    """Uses a simple NN to predict an optimal stock/bond mix. Returns standard Python floats."""
    
    # Train a dummy model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(2,)),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Dummy training data
    X = np.array([[5, 0], [10, 1], [20, 2]])
    y = np.array([[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]])
    model.fit(X, y, epochs=10, verbose=0)
    
    risk_code = {'low':0, 'medium':1, 'high':2}.get(risk, 1)
    
    pred = model.predict(np.array([[years, risk_code]]))[0]
    
    # Ensures return values are standard Python floats for BSON compatibility
    return {
        'stocks': float(pred[0]), 
        'bonds': float(pred[1])
    }
