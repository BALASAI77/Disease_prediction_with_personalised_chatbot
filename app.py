import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import google.generativeai as genai
import time


# Load data
df2 = pd.read_csv('Symptom-severity.csv')
df = pd.read_csv('dataset.csv')

# Function to predict disease
def predict_disease(symptom_selections):
    # Concatenate 'Symptom_1' to 'Symptom_5' columns
    all_symptoms = pd.concat([df['Symptom_1'], df['Symptom_2'], df['Symptom_3'], df['Symptom_4'], df['Symptom_5']])

    # Drop NaN values and get unique symptoms
    unique_symptoms = all_symptoms.dropna().unique()

    # Create a DataFrame with unique symptoms as columns
    unique_symptoms_df = pd.DataFrame(columns=unique_symptoms)
    unique_symptoms_df.rename(columns=lambda x: x.strip(), inplace=True)

    for index, row in df.iterrows():
        symptom_values = []
        for symptom in unique_symptoms:
            if symptom in row.values:
                symptom_values.append(1)
            else:
                symptom_values.append(0)
        unique_symptoms_df.loc[index] = symptom_values

    target = df['Disease']
    one_hot_encoded_target = pd.get_dummies(target)
    X_train, X_test, y_train, y_test = train_test_split(unique_symptoms_df, one_hot_encoded_target, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    prediction_df = pd.DataFrame(columns=unique_symptoms)
    symptom_selections_ordered = [symptom for symptom in prediction_df.columns if symptom in symptom_selections]
    
    prediction_df.loc[0, symptom_selections_ordered] = 1
    prediction_df.fillna(0, inplace=True)
    prediction_df.rename(columns=lambda x: x.strip(), inplace=True)

    X_prediction = prediction_df
    predicted_disease = clf.predict(X_prediction)
    predicted_disease_column = one_hot_encoded_target.columns[predicted_disease.argmax()]

    return predicted_disease_column

# Function to generate response using Gemini API
def generate_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response

# Define a session state to store generated response


st.title("Symptom Selector")

symptom_df = df2['Symptom']
symptom_selections = []
for i in range(5):
    symptom_selections.append(st.selectbox(f"Symptom {i+1}", symptom_df, key=f"symptom_{i}"))

submitted = st.button("Submit")

if submitted:
    st.write("Symptoms Selected:")
    st.write(symptom_selections)

    # Predict disease
    predicted_disease_column = predict_disease(symptom_selections)
    st.write("Predicted Disease:")
    st.write(predicted_disease_column)

    # Generate responses
    prompt_diet = f'You need to now act as a doctor. A person came to you and he is tested positive with a disease called {predicted_disease_column}. Your job is to give him the diet plan and some fitness suggestions and some of the home remedies to recover fast from that disease. Also make it more accurate with respect to that disease.'

    diet_response = generate_response(prompt_diet)
    time.sleep(2)
    total_severity = df2[df2['Symptom'].isin(symptom_selections)]['weight'].sum()
    average_severity = total_severity / len(symptom_selections)
    severity_percentage=(average_severity/7)*100
    inner_div_color = f'linear-gradient(0deg, red {severity_percentage}%, transparent {severity_percentage}%)'
    st.title("Disease Severity Bar")
    #st.write(total_severity)
    #st.write(average_severity)
    st.write(f'The severity percentage of {predicted_disease_column} is : {severity_percentage}')
    #st.write(severity_percentage)
    st.markdown(
        f"""
        <div style="background-color: #f0f0f0; width: 30px; height: 200px; position: relative;">
            <div style="background: {inner_div_color}; width: 100%; height: 100%; position: absolute; bottom: 0;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )




    st.title("Diet, Fitness and Home Remedies suggestions :")
    st.write(diet_response.text)



# Display responses





