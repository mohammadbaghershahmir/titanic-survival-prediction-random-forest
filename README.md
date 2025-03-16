# Titanic Survival Prediction using Random Forest

## 📌 Overview
This project applies a **Random Forest classification model** to predict whether a passenger survived the Titanic disaster. The model is trained on the Titanic dataset after performing preprocessing and encoding of categorical features.

Key aspects include:
- Handling missing values in the 'Age' column
- Encoding categorical variables using `LabelEncoder`
- Model training using `RandomForestClassifier`
- Evaluation via F1 Score and survival count comparison

## 📊 Technologies Used
- Python
- Scikit-learn
- Pandas
- LabelEncoder

## 📁 Project Structure
```
data/
    titanic.csv               → Dataset file (add it manually)
```

## ▶️ How to Run

1. Clone the repository:
```
git clone https://github.com/mohammadbaghershahmir/titanic-survival-prediction-random-forest.git
```

2. Add the `titanic.csv` file into the `data/` folder.

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the script:
```bash
cd src
python Q22.py
```

## 📈 Sample Output
- F1 Score
- Count of real survivors vs predicted survivors

## 🏷️ Tags
`machine-learning` `random-forest` `classification` `titanic` `survival-prediction` `scikit-learn`

## 📄 License
MIT
