# Cardiovascular Risk Predictor

## Introduction
This Flask-based web application, "Cardiovascular Risk Predictor," evaluates the potential risk of cardiovascular diseases based on a variety of health metrics. It is powered by fuzzy logic, which allows for nuanced decision-making based on uncertain or imprecise data. This application is particularly useful for individuals seeking to understand their cardiovascular health risks based on personal health data.

## Features
- **Individual Risk Assessment**: Users can input personal health data to receive an assessment of their cardiovascular risk.
- **Fuzzy Logic Decision Making**: Utilizes fuzzy logic for more accurate assessments based on complex rule-based calculations.
- **Health Recommendations**: Based on the assessed risk, the application provides personalized health advice to help users manage or reduce their risk.

## Tech Stack
- **Flask**: A micro web framework written in Python, used for backend development.
- **Python**: The core programming language used for developing the logic and functionalities of the application.
- **HTML/CSS**: Used for creating and styling the web interface.
- **Skfuzzy**: A Python library that supports building systems with fuzzy logic.

## Project Structure
```plaintext
├── main.py           # Main Flask application file containing all routes and logic.
├── templates/        # Folder containing HTML files.
│   └── index.html    # HTML file for the web interface.
├── static/           # Folder containing CSS and potentially other static files.
│   └── style.css     # CSS file for styling the web interface.
└── tests.txt         # File containing test data samples for demonstration purposes
```


# Setup
To get the application up and running on your local machine, follow these steps:

##Clone the Repository:
```bash
Copier le code
git clone <repository-url>  # Replace <repository-url> with the URL of your GitHub repository.
```
##Install Dependencies:
```bash
Copier le code
pip install flask numpy pandas skfuzzy networkx
```
##Run the Application:
```bash
Copier le code
cd path/to/directory  # Change directory to where you cloned the repo.
python main.py        # Execute the main application file.
```
#Usage
##Accessing the Web Interface
Open a web browser and go to http://127.0.0.1:5000/ to interact with the application.

##Entering Data
Input your health data in the provided form fields:
Age, Gender, Height, Weight
Systolic and Diastolic Blood Pressure
Cholesterol and Glucose Levels
Lifestyle factors such as smoking and alcohol consumption
Receiving Predictions
Submit the form to see your cardiovascular risk level displayed along with tailored health recommendations.

#Contributing
Contributions to this project are welcome. To contribute:

#Fork the repository.
Create a new branch for your feature 
```git checkout -b feature/AmazingFeature.```
```git commit -m 'Add some AmazingFeature.'```
```git push origin feature/AmazingFeature.```
Open a pull request.
#License
This project is made available under the MIT License. For more details, see the LICENSE.md file included with the repository.

```
This detailed README provides a comprehensive guide to your project, suitable for inclusion in a GitHub repository.
It helps potential users and contributors understand how to install, use, and contribute to the project effectively.
If there's anything else you'd like to add or modify, let me know!
```

