import numpy as np
import pandas as pd
import joblib
import streamlit as st

#Load the model
model = joblib.load('model')

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.drop(['id'], axis = 1, inplace = True)

def predict_car_price(input):
    prediction = model.predict_proba(input)
    return prediction[0], np.argmax(prediction[0])

def main():

    with st.sidebar.beta_expander('About'):
        st.write('This Machine learning app uses Logistic Regression to predict whether a person may get stroke or not based on the inputs. The front end is built using Streamlit.')

    with st.sidebar.beta_expander('Contact'):
        st.write('[GitHub](https://github.com/VaisakNair7/Stroke-Prediction)')
        st.write('[LinkedIn](https://www.linkedin.com/in/vaisaksnair/)')
        st.write('Mail : vaisaksnair98@gmail.com')
    
    html_temp = """
    <div style="background-color:#DD4124;padding:2px">
    <h1 style="color:white;text-align:center;">Stroke Prediction</h1>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    #Get inputs from user.

    gender = st.selectbox('Gender', ['Male', 'Female'])
    
    age = st.number_input('Age', min_value = 1, step = 1, value = 30)

    hypertension = st.selectbox('Do you have hypertension?', ['Yes', 'No'])
    if hypertension == 'Yes':
        hypertension = 1
    else:
        hypertension = 0

    heart_disease = st.selectbox('Do you have any heart disease?', ['Yes', 'No'])
    if heart_disease == 'Yes':
        heart_disease = 1
    else:
        heart_disease = 0

    married = st.selectbox('Are you married?', ['Yes', 'No'])

    work_type = st.selectbox('Type of work you do', ['Private', 'Self Employed', 'Government Job', 'Children', 'Never worked'])
    if married == 'Self Employed':
        married = 'Self-employed'
    elif married == 'Government Job':
        married = 'Govt_job'
    elif married == 'Children':
        married = 'children'
    elif married == 'Never worked':
        married = 'Never_worked'

    residence = st.selectbox('Residence type', ['Urban', 'Rural'])

    glucose = st.number_input('Glucose Level', min_value = 1, step = 1, value = 100)

    bmi = st.number_input('BMI', min_value = 1, step = 1, value = 23)

    smoking = st.selectbox('Smoking status', ['Never Smoked', 'Unknown/Information unavailable', 'Formerly smoked', 'Smokes'])
    if smoking == 'Never Smoked':
        smoking = 'never smoked'
    elif smoking == 'Formerly smoked':
        smoking = 'formerly smoked'
    elif smoking == 'Smokes':
        smoking = 'smokes'
    elif smoking == 'Unknown/Information unavailable':
        smoking = 'Unknown'
    
    #Prediction
    if st.button('Predict'):
        input = pd.DataFrame([[gender, age, hypertension, heart_disease, married, work_type, residence, glucose, bmi, smoking]], columns = df.columns[:-1])
        prob, stroke = predict_car_price(input)
        if stroke:
            st.error('YOU ARE {}% LIKELY TO GET STROKE 😓'.format(np.round(prob[1] * 100, 1)))
        else:
            st.success('YOU ARE {}% UNLIKELY TO GET STROKE 😇'.format(np.round(prob[0 ] * 100, 1)))
    

if __name__ == '__main__':
    main()

