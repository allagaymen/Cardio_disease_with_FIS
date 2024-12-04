from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import networkx as nx
import random
import os
from skfuzzy import control as ctrl

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

age = ctrl.Antecedent(np.arange(18, 101, 1), 'age')
gender = ctrl.Antecedent(np.arange(1, 3, 1), 'gender')  # Assuming 1 for female and 2 for male
height = ctrl.Antecedent(np.arange(120, 221, 1), 'height')
weight = ctrl.Antecedent(np.arange(30, 201, 1), 'weight')
ap_high = ctrl.Antecedent(np.arange(90, 241, 1), 'ap_high')
ap_low = ctrl.Antecedent(np.arange(60, 181, 1), 'ap_low')
cholesterol = ctrl.Antecedent(np.arange(1, 4, 1), 'cholesterol')  # Assuming 1, 2, 3 are categorical values for cholesterol levels
glucose = ctrl.Antecedent(np.arange(1, 4, 1), 'glucose')  # Assuming 1, 2, 3 are categorical values for glucose levels
smoke = ctrl.Antecedent(np.arange(0, 2, 1), 'smoke')
alcohol = ctrl.Antecedent(np.arange(0, 2, 1), 'alcohol')
physical_activity = ctrl.Antecedent(np.arange(0, 2, 1), 'physical_activity')

# Consequent (Output)
cardio_disease = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'cardio_disease')

age['young'] = fuzz.trimf(age.universe, [18, 18, 20])
age['middle_aged'] = fuzz.trimf(age.universe, [21, 50, 70])
age['senior'] = fuzz.trimf(age.universe, [60, 85, 100])

# Example Membership Functions for Cholesterol (you would define these similarly for other variables)
cholesterol['low'] = fuzz.trimf(cholesterol.universe, [1, 1, 2])
cholesterol['moderate'] = fuzz.trimf(cholesterol.universe, [1, 2, 3])
cholesterol['high'] = fuzz.trimf(cholesterol.universe, [2, 3, 3])

# Membership functions for Systolic Blood Pressure (ap_high)
ap_high['normal'] = fuzz.trimf(ap_high.universe, [90, 90, 120])
ap_high['elevated'] = fuzz.trimf(ap_high.universe, [90, 120, 130])
ap_high['high'] = fuzz.trimf(ap_high.universe, [120, 130, 240])

# Membership functions for Diastolic Blood Pressure (ap_low)
ap_low['normal'] = fuzz.trimf(ap_low.universe, [60, 60, 80])
ap_low['elevated'] = fuzz.trimf(ap_low.universe, [60, 80, 90])
ap_low['high'] = fuzz.trimf(ap_low.universe, [80, 90, 180])

gender['female'] = fuzz.trimf(gender.universe, [1, 1, 1.5])
gender['male'] = fuzz.trimf(gender.universe, [1.5, 2, 2])

height['short'] = fuzz.trimf(height.universe, [120, 120, 165])
height['average'] = fuzz.trimf(height.universe, [155, 175, 195])
height['tall'] = fuzz.trimf(height.universe, [185, 220, 220])

weight['underweight'] = fuzz.trimf(weight.universe, [30, 30, 55])
weight['normal'] = fuzz.trimf(weight.universe, [50, 70, 90])
weight['overweight'] = fuzz.trimf(weight.universe, [85, 200, 200])

smoke['non_smoker'] = fuzz.trimf(smoke.universe, [0, 0, 0.5])
smoke['smoker'] = fuzz.trimf(smoke.universe, [0.5, 1, 1])

alcohol['non_drinker'] = fuzz.trimf(alcohol.universe, [0, 0, 0.5])
alcohol['drinker'] = fuzz.trimf(alcohol.universe, [0.5, 1, 1])

physical_activity['inactive'] = fuzz.trimf(physical_activity.universe, [0, 0, 0.5])
physical_activity['active'] = fuzz.trimf(physical_activity.universe, [0.5, 1, 1])


glucose['normal'] = fuzz.trimf(glucose.universe, [1, 1, 2])
glucose['pre_high'] = fuzz.trimf(glucose.universe, [1, 2, 3])
glucose['high'] = fuzz.trimf(glucose.universe, [2, 3, 3])

# Consequent (Output) with updated membership functions
cardio_disease['low'] = fuzz.trimf(cardio_disease.universe, [0, 0, 0.5])
cardio_disease['medium'] = fuzz.trimf(cardio_disease.universe, [0, 0.5, 1])
cardio_disease['high'] = fuzz.trimf(cardio_disease.universe, [0.5, 1, 1])

# Create the Fuzzy Control System
cardio_control_system = ctrl.ControlSystem()

# Rule 1: High risk of cardiovascular disease based on several high risk factors
rule1 = ctrl.Rule(cholesterol['high'] | glucose['high'] | ap_high['high'], cardio_disease['high'])

# Rule 2: Moderate risk due to moderate levels of cholesterol or glucose, especially with elevated blood pressure
rule2 = ctrl.Rule((cholesterol['moderate'] | glucose['pre_high']) & (ap_high['elevated'] | ap_low['elevated']), cardio_disease['medium'])

# Rule 3: Low risk due to normal measurements and healthy lifestyle
rule3 = ctrl.Rule((cholesterol['low'] & glucose['normal']) & (ap_high['normal'] & ap_low['normal']) & physical_activity['active'], cardio_disease['low'])

# Rule 4: High risk for smokers and drinkers with elevated or high cholesterol/glucose
rule4 = ctrl.Rule((smoke['smoker'] | alcohol['drinker']) & (cholesterol['high'] | glucose['high']), cardio_disease['high'])

# Rule 5: Medium risk for overweight individuals with moderate to high cholesterol or blood pressure
rule5 = ctrl.Rule(weight['overweight'] & (cholesterol['moderate'] | ap_high['elevated']), cardio_disease['medium'])

# Rule 6: Gender-specific risk adjustment
rule6 = ctrl.Rule(gender['male'] & (ap_high['high'] | cholesterol['high']), cardio_disease['high'])

rule7 = ctrl.Rule(gender['female'] & physical_activity['inactive'] & (cholesterol['moderate'] | glucose['pre_high']), cardio_disease['medium'])

# Rule 8: High risk when both blood pressures are high, especially with smoking or alcohol consumption
rule8 = ctrl.Rule((ap_high['high'] & ap_low['high']) & (smoke['smoker'] | alcohol['drinker']), cardio_disease['high'])

# Rule 9: Moderate risk for seniors with elevated blood pressure or borderline glucose
rule9 = ctrl.Rule(age['senior'] & (ap_high['elevated'] | glucose['pre_high']), cardio_disease['medium'])

# Rule 10: Low risk for young, active individuals with normal weight and no smoking or alcohol habits
rule10 = ctrl.Rule(age['young'] & physical_activity['active'] & weight['normal'] & smoke['non_smoker'] & alcohol['non_drinker'], cardio_disease['low'])

# Rule 11: Medium risk for individuals with moderate cholesterol and glucose levels, without active lifestyle
rule11 = ctrl.Rule((cholesterol['moderate'] & glucose['pre_high']) & physical_activity['inactive'], cardio_disease['medium'])

# Rule 12: High risk for overweight smokers or drinkers with high cholesterol
rule12 = ctrl.Rule(weight['overweight'] & (smoke['smoker'] | alcohol['drinker']) & cholesterol['high'], cardio_disease['high'])

# Rule 13: Medium risk for females with elevated cholesterol or glucose who consume alcohol
rule13 = ctrl.Rule(gender['female'] & (cholesterol['moderate'] | glucose['pre_high']) & alcohol['drinker'], cardio_disease['medium'])

# Rule 14: Low risk for males with all health indicators normal and active lifestyle
rule14 = ctrl.Rule(gender['male'] & cholesterol['low'] & glucose['normal'] & ap_high['normal'] & ap_low['normal'] & physical_activity['active'], cardio_disease['low'])

# Rule 15: High risk for any individual with high glucose and high blood pressure, regardless of other factors
rule15 = ctrl.Rule(glucose['high'] & (ap_high['high'] | ap_low['high']), cardio_disease['high'])
# Rule 16: High risk for short individuals with high cholesterol or glucose
rule16 = ctrl.Rule(height['short'] & (cholesterol['high'] | glucose['high']), cardio_disease['high'])

# Rule 17: Medium risk for individuals with average height and elevated cholesterol or glucose
rule17 = ctrl.Rule(height['average'] & (cholesterol['moderate'] | glucose['pre_high']), cardio_disease['medium'])

# Rule 18: Low risk for tall individuals with normal cholesterol, glucose, and active lifestyle
rule18 = ctrl.Rule(height['tall'] & cholesterol['low'] & glucose['normal'] & physical_activity['active'], cardio_disease['low'])

# Rule 19: High risk for short individuals who are smokers or drinkers, with high cholesterol
rule19 = ctrl.Rule(height['short'] & (smoke['smoker'] | alcohol['drinker']) & cholesterol['high'], cardio_disease['high'])

# Rule 20: Medium risk for average height individuals with moderate cholesterol and glucose, and inactive lifestyle
rule20 = ctrl.Rule(height['average'] & (cholesterol['moderate'] | glucose['pre_high']) & physical_activity['inactive'], cardio_disease['medium'])

# Rule 21: High risk for tall individuals with high blood pressure and high cholesterol or glucose
rule21 = ctrl.Rule(height['tall'] & (ap_high['high'] | ap_low['high']) & (cholesterol['high'] | glucose['high']), cardio_disease['high'])

# Rule 22: Medium risk for short individuals with elevated blood pressure and moderate cholesterol
rule22 = ctrl.Rule(height['short'] & (ap_high['elevated'] | ap_low['elevated']) & cholesterol['moderate'], cardio_disease['medium'])

# Rule 23: Low risk for tall individuals with no smoking, alcohol, or cholesterol problems
rule23 = ctrl.Rule(height['tall'] & smoke['non_smoker'] & alcohol['non_drinker'] & cholesterol['low'], cardio_disease['low'])

# Rule 24: High risk for individuals of average height with high cholesterol and an inactive lifestyle
rule24 = ctrl.Rule(height['average'] & cholesterol['high'] & physical_activity['inactive'], cardio_disease['high'])

# Rule 25: Medium risk for short individuals with a borderline high blood pressure and pre-high glucose levels
rule25 = ctrl.Rule(height['short'] & (ap_high['elevated'] | ap_low['elevated']) & glucose['pre_high'], cardio_disease['medium'])

# Rule 26: High risk for tall individuals with high blood pressure and high glucose levels, regardless of other factors
rule26 = ctrl.Rule(height['tall'] & (ap_high['high'] | ap_low['high']) & glucose['high'], cardio_disease['high'])

# Rule 27: Low risk for young, tall individuals with normal cholesterol, glucose, and active lifestyle
rule27 = ctrl.Rule(age['young'] & height['tall'] & cholesterol['low'] & glucose['normal'] & physical_activity['active'], cardio_disease['low'])

# Rule 28: Medium risk for average height individuals with elevated blood pressure and high cholesterol or glucose
rule28 = ctrl.Rule(height['average'] & (ap_high['elevated'] | ap_low['elevated']) & (cholesterol['moderate'] | glucose['pre_high']), cardio_disease['medium'])



# Adding rules to control system
cardio_control_system.addrule(rule1)
cardio_control_system.addrule(rule2)
cardio_control_system.addrule(rule3)
cardio_control_system.addrule(rule4)
cardio_control_system.addrule(rule5)
cardio_control_system.addrule(rule6)
cardio_control_system.addrule(rule7)
cardio_control_system.addrule(rule8)
cardio_control_system.addrule(rule9)
cardio_control_system.addrule(rule10)
cardio_control_system.addrule(rule11)
cardio_control_system.addrule(rule12)
cardio_control_system.addrule(rule13)
cardio_control_system.addrule(rule14)
cardio_control_system.addrule(rule15)
cardio_control_system.addrule(rule16)
cardio_control_system.addrule(rule17)
cardio_control_system.addrule(rule18)
cardio_control_system.addrule(rule19)
cardio_control_system.addrule(rule20)
cardio_control_system.addrule(rule21)
cardio_control_system.addrule(rule22)
cardio_control_system.addrule(rule23)
cardio_control_system.addrule(rule24)
cardio_control_system.addrule(rule25)
cardio_control_system.addrule(rule26)
cardio_control_system.addrule(rule27)
cardio_control_system.addrule(rule28)

# Create a Control System Simulation to run our controller
cardio_simulator = ctrl.ControlSystemSimulation(cardio_control_system)
def compute_cardio_risk(AGE, GENDER, HEIGHT, WEIGHT, AP_HIGH, AP_LOW, CHOLESTEROL, GLUCOSE, SMOKE, ALCOHOL, PHYSICAL_ACTIVITY):
    cardio_simulator.input['age'] = AGE
    cardio_simulator.input['gender'] = GENDER
    cardio_simulator.input['height'] = HEIGHT
    cardio_simulator.input['weight'] = WEIGHT
    cardio_simulator.input['ap_high'] = AP_HIGH
    cardio_simulator.input['ap_low'] = AP_LOW
    cardio_simulator.input['cholesterol'] = CHOLESTEROL
    cardio_simulator.input['glucose'] = GLUCOSE
    cardio_simulator.input['smoke'] = SMOKE
    cardio_simulator.input['alcohol'] = ALCOHOL
    cardio_simulator.input['physical_activity'] = PHYSICAL_ACTIVITY
    cardio_simulator.compute()
    return cardio_simulator.output['cardio_disease']


@app.route("/", methods=["GET", "POST"])
def individual():
    # Define advice messages for each risk class
    advice_messages = {
    "low": (
        "**Your Cardiovascular Risk: Low"
        "This indicates that your current lifestyle and health indicators are generally supportive of cardiovascular health. "
        "However, maintaining this status requires continued attention to preventive measures.\n\n"
        "**Recommendations**\n"
        "1. **Healthy Diet**\n"
        "   - Consume a balanced diet rich in:\n"
        "     - Fruits\n"
        "     - Vegetables\n"
        "     - Whole grains\n"
        "     - Lean proteins\n"
        "     - Healthy fats\n"
        "   - Limit intake of:\n"
        "     - Salt\n"
        "     - Sugar\n"
        "     - Saturated fats\n\n"
        "2. **Physical Activity**\n"
        "   - Engage in at least:\n"
        "     - 150 minutes of moderate-intensity exercise **or**\n"
        "     - 75 minutes of vigorous activity weekly.\n\n"
        "3. **Regular Check-ups**\n"
        "   - Schedule periodic health assessments to monitor:\n"
        "     - Blood pressure\n"
        "     - Cholesterol levels\n"
        "     - Blood glucose\n\n"
        "4. **Avoid Risky Behaviors**\n"
        "   - Refrain from smoking.\n"
        "   - Limit alcohol consumption to maintain your low-risk status.\n\n"
        "5. **Stress Management**\n"
        "   - Incorporate stress-reducing practices such as:\n"
        "     - Mindfulness\n"
        "     - Yoga\n"
        "     - Relaxation techniques\n"
    ),
    "medium": (
        "**Your Cardiovascular Risk: Medium**\n\n"
        "Your risk level is moderate. It’s essential to start taking preventive measures now to lower your risk. "
        "Small lifestyle changes can make a significant difference over time.\n\n"
        "**Recommendations**\n"
        "1. **Healthy Diet**\n"
        "   - Adopt a diet low in salt, sugar, and saturated fats.\n"
        "   - Increase intake of fruits, vegetables, and whole grains.\n\n"
        "2. **Regular Exercise**\n"
        "   - Aim for at least 30 minutes of physical activity, 5 times a week.\n\n"
        "3. **Weight Management**\n"
        "   - Maintain a healthy weight to reduce strain on your heart.\n\n"
        "4. **Health Monitoring**\n"
        "   - Regularly check your blood pressure, cholesterol, and blood sugar levels.\n\n"
        "5. **Consult Your Doctor**\n"
        "   - Discuss potential risk factors and preventive measures tailored to your health profile.\n"
    ),
    "high": (
        "Your Cardiovascular Risk: High**\n\n"
        "This indicates a significant risk to your heart health. Immediate action is required to reduce your risk and prevent complications.\n\n"
        "**Recommendations**\n"
        "1. **Medical Consultation**\n"
        "   - Consult a healthcare provider for a comprehensive evaluation and personalized treatment plan.\n\n"
        "2. **Healthy Lifestyle Changes**\n"
        "   - Avoid smoking and alcohol completely.\n"
        "   - Adopt a heart-healthy diet rich in fruits, vegetables, and low-fat proteins.\n\n"
        "3. **Medication**\n"
        "   - Follow your doctor’s advice regarding medication to control blood pressure, cholesterol, or other conditions.\n\n"
        "4. **Regular Monitoring**\n"
        "   - Monitor key health metrics frequently, such as blood pressure, cholesterol, and glucose levels.\n\n"
        "5. **Stay Active**\n"
        "   - Include low-impact physical activities like walking or swimming, as advised by your doctor.\n"
        "   - Avoid strenuous activities without medical clearance.\n\n"
        "6. **Stress Management**\n"
        "   - Engage in stress-reducing practices, such as meditation or therapy, to support your overall health.\n"
    )
}


    
    # Corrected input assignments without commas
    AGE = int(request.form['age'])
    GENDER = int(request.form['gender'])
    HEIGHT = int(request.form['height'])
    WEIGHT = int(request.form['weight'])
    AP_HIGH = int(request.form['ap_high'])
    AP_LOW = int(request.form['ap_low'])
    CHOLESTEROL = int(request.form['cholesterol'])
    GLUCOSE = int(request.form['glucose'])
    SMOKE = int(request.form['smoke'])
    ALCOHOL = int(request.form['alcohol'])
    PHYSICAL_ACTIVITY = int(request.form['physical_activity'])
    
    # Call the function with the correct input values
    risk = compute_cardio_risk(AGE, GENDER, HEIGHT, WEIGHT, AP_HIGH, AP_LOW, CHOLESTEROL, GLUCOSE, SMOKE, ALCOHOL, PHYSICAL_ACTIVITY)
    
    # Return the formatted risk percentage
    risk_percent = round(risk*100, 2)
    risk = f"{risk_percent:.2f}%"
    # Classify the risk level based on the percentage
    if risk_percent <= 33:
        risk_class = "low"
    elif 33 < risk_percent <= 66:
        risk_class = "medium"
    else:
        risk_class = "high"

        # Get the corresponding advice
    advice = advice_messages[risk_class]

    # Render the template with risk, risk_class, and advice
    return render_template("index.html", risk=risk, risk_class=risk_class, advice=advice)
# @app.route("/batch", methods=["POST", "GET"])
# def batch():
#     if request.method == "POST":
#         if "csvfile" not in request.files:
#             return "No file part", 400

#         file = request.files["csvfile"]
#         if file.filename == "":
#             return "No selected file", 400

#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         try:
#             df = pd.read_csv(filepath)
#             df['Predicted_Risk'] = df.apply(compute_cardio_risk, axis=1)
#             df['Predicted_Risk'] = df['Predicted_Risk'].apply(lambda x: f"{x:.2f}%" if x is not None else "Error")

#             output_filename = f"processed_{file.filename}"
#             output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
#             df.to_csv(output_filepath, index=False)
#             return render_template("batch.html", filename=f"processed/{output_filename}")
#         except Exception as e:
#             print(f"Error processing file: {e}")
#             return "Error processing file", 500

#     return render_template("batch.html")

if __name__ == "__main__":
    app.run(debug=True)
