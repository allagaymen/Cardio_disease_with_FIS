# Cardiovascular Risk Predictor

A Flask web app that estimates a person's risk of cardiovascular disease from 11 health details, using a fuzzy inference system instead of a trained model.

## How it works

- A web form asks for age, gender, height, weight, blood pressure (systolic and diastolic), cholesterol and glucose levels (1 to 3), and whether the person smokes, drinks alcohol or is physically active.
- Each value is mapped to fuzzy sets with `scikit-fuzzy`, such as young, middle aged or senior, and normal, elevated or high blood pressure.
- 28 hand-written rules combine these sets. For example, high cholesterol, high glucose or high systolic pressure leads to high risk.
- The system returns a risk score from 0 to 100%, labelled low (up to 33%), medium (up to 66%) or high.
- The page shows the score with general health advice for that risk level.

`tests.txt` has sample inputs to try. They are taken from `cardiovascular_diseases_dv3.csv`, which also includes the real outcome for each person.

## Tech

- Python, Flask
- scikit-fuzzy, NumPy, pandas, NetworkX
- HTML and CSS

## Run it locally

```bash
pip install flask numpy pandas scikit-fuzzy networkx
python main.py
```

Then open http://127.0.0.1:5000/ in a browser.

This is a student project and not a medical tool.

---

Author: Aymen Allag ([allagaymen](https://github.com/allagaymen)) · README by Ahmed Amine Tiffrent ([portfolio](https://www.aminetiffrent.com/portfolio))
