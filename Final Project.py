import pandas as pd
import numpy as np 
import streamlit as st
import seaborn as sns 
import sklearn as sk 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Linkedin User Predictor")
st.write("The purpose of this application is to predict whether or not an individual is likely to be a LinkedIn user based on a set of provided parameters.")

s = pd.read_excel('C:\\Users\\camer\\OneDrive\\Desktop\\Georgetown MSB\\Fall 2023\\Module 2\Programming II\\social_media_usage.xlsx')

def clean_sm(x):
    return np.where(x == 1,1,0)

data = {'Column 1': [1,0,1], 'Column2': [0,1,1]}
toy_df = pd.DataFrame(data)
cleaned_df = toy_df.apply(clean_sm)
print(toy_df)
print(cleaned_df)
s_data = {
    "sm_li": s["web1h"],
    "income": s["income"],
    "education": s["educ2"],
    "parent": s["par"],
    "married": s["marital"],
    "female": s["gender"],
    "age": s["age"]
}

ss = pd.DataFrame(s_data)
ss["sm_li"] = clean_sm(ss["sm_li"])
ss["female"] = np.where(ss["female"] == 2, 1, 0)
ss["married"] = np.where(ss["married"] == 1, 1, 0)
ss["parent"] = np.where(ss["parent"] == 1, 1, 0)
ss["income"] = np.where(ss["income"] > 9, np.nan, ss["income"])
ss["age"] = np.where(ss["age"] > 98, np.nan, ss["age"])
ss["education"] = np.where(ss["education"] > 8, np.nan, ss["education"])
ss.dropna(subset=["income", "age", "education"], inplace=True)

#Income Variable 
income_ranges = [
    '< $10,000',
    '$10,000 - $19,999',
    '$20,000 - $29,999',
    '$30,000 - $49,999',
    '$50,000 - $79,999',
    '$80,000 - $99,999',
    '$100,000 - $149,999',
    '$150,000 - 199,999',
    '> $200,000'
]

income_mapping = {
    '< $10,000': 1,
    '$10,000 - $19,999': 2,
    '$20,000 - $29,999': 3,
    '$30,000 - $49,999': 4,
    '$50,000 - $79,999': 5,
    '$80,000 - $99,999': 6,
    '$100,000 - $149,999': 7,
    '$150,000 - $199,999': 8,
    '> $200,000': 9
}

selected_income_range = st.selectbox('Select an Income Category:', income_ranges)

income = income_mapping[selected_income_range]

#Education Variable
education_categories = [
    'Elementary School Complete',
    'Middle School Complete',
    'High School Complete',
    'Attended College, did not graduate',
    'Vocational School/Associate"s Degree',
    'Bachelor"s Degree',
    'Master"s Degree',
    'Doctorate Level Degree',
]

education_mapping = {
    'Elementary School Complete': 1,
    'Middle School Complete': 2,
    'High School Complete': 3,
    'Attended College, did not graduate': 4,
    'Vocational School/Associate"s Degree': 5,
    'Bachelor"s Degree': 6,
    'Master"s Degree': 7,
    'Doctorate Level Degree': 8
}

selected_education = st.selectbox('Select an education level:', education_categories)

education = education_mapping[selected_education]

#Age Variable
age = st.number_input('Enter your age, Please enter a number between 1 and 98:', min_value=1, max_value=98, value=1, step=1)

#Parent Variable
parental_status_options = ['Not a parent', 'Parent']
selected_parental_status = st.radio('Are you a parent?', parental_status_options)
parent = 1 if selected_parental_status == 'Parent' else 0

#Gender Variable
gender_options = ['Male', 'Female']
selected_gender = st.radio('Select your gender:', gender_options)
female = 1 if selected_gender == 'Female' else 0

#Marital Status Variable
marital_status_options = ['Single', 'Married']
selected_marital_status = st.radio('Are you married?', marital_status_options)
married = 1 if selected_marital_status == 'Married' else 0


y = ss['sm_li']
x = ss.drop('sm_li', axis = 1)
logreg_model = LogisticRegression(class_weight='balanced')
model_input_data = pd.DataFrame({'income': [income], 'education': [int(education)], 'parent':[int(parent)], 'married':[int(married)], "female":[int(female)], 'age':[int(age)]})

logreg_model.fit(x, y)
user_prediction = logreg_model.predict(model_input_data)
user_probability = logreg_model.predict_proba(model_input_data)[:,1]

st.write(f'Our model predicts that you are most likely a LinkedIn user' if user_prediction == 1 else 'Our model predicts that you are most likely not a LinkedIn user')
st.write(f'Our model suggest that the probability of you being a LinkedIn user based on your inputs is: {user_probability}')

if __name__ == "__main__":
    main()
