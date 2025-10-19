import datetime
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId # Import ObjectId for clarity
import os
# Assumes finance_utils.py contains the corrected functions
from finance_utils import calculate_contributions, run_monte_carlo, predict_optimal_mix 

app = Flask(__name__)

# MongoDB connection
# NOTE: Replace 'localhost:27017' with your actual MongoDB connection string if running remotely
client = MongoClient('mongodb://localhost:27017/') 
db = client['financial_db']
goals_collection = db['goals']

# Helper function to determine the expected return rate based on risk
def get_expected_rate(risk_tolerance):
    """Maps risk tolerance to an expected annual return rate for contribution estimation."""
    if risk_tolerance == 'low': return 0.03
    elif risk_tolerance == 'medium': return 0.05
    elif risk_tolerance == 'high': return 0.07
    return 0.05 # Default 

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Extract form data and ensure types are correct
        goal_type = request.form['goal_type']
        target_amount = float(request.form['target_amount'])
        years = int(request.form['years']) # The variable is correctly extracted here
        current_savings = float(request.form['current_savings'])
        risk_tolerance = request.form['risk_tolerance']  # 'low', 'medium', 'high'

        # Get the appropriate expected rate for the contribution calculation
        expected_rate = get_expected_rate(risk_tolerance)

        # 1. Calculate Monthly Contribution 
        monthly_contrib = calculate_contributions(target_amount, years, current_savings, expected_rate)
        
        # 2. Run Monte Carlo Simulation 
        sim_results = run_monte_carlo(target_amount, years, monthly_contrib, current_savings, risk_tolerance)
        
        # 3. Predict Investment Mix 
        investment_mix = predict_optimal_mix(years, risk_tolerance)

        # Store in MongoDB 
        # Note: BSON compatibility is handled in finance_utils by converting NumPy types to float/dict
        goal_doc = {
            'timestamp': datetime.datetime.utcnow(),
            'goal_type': goal_type,
            'target_amount': target_amount,
            'years': years,
            'current_savings': current_savings,
            'risk_tolerance': risk_tolerance,
            'monthly_contrib': monthly_contrib,
            'investment_mix': investment_mix,
            'success_prob': sim_results['success_prob'],
            'median_wealth': sim_results['median_wealth'] 
        }
        goal_id = goals_collection.insert_one(goal_doc).inserted_id

        # FIX: The 'years' variable is now explicitly passed to the template
        return render_template('results.html', 
                                 monthly_contrib=monthly_contrib, 
                                 investment_mix=investment_mix, 
                                 sim_results=sim_results,
                                 years=years, 
                                 risk_tolerance=risk_tolerance, # <--- THIS IS THE FIX
                                 goal_id=str(goal_id))

    # Placeholder for rendering the input form template
    return render_template('index.html')

@app.route('/goal/<goal_id>', methods=['GET'])
def view_goal(goal_id):
    try:
        oid = ObjectId(goal_id)
    except:
        return 'Invalid Goal ID format', 400
        
    goal = goals_collection.find_one({'_id': oid}) 
    if goal:
        goal['_id'] = str(goal['_id'])
        return jsonify(goal)
    return 'Goal not found', 404

if __name__ == '__main__':
    # Ensure MongoDB is running before executing
    app.run(debug=True)
