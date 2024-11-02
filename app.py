import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model

# Set the page configuration
st.set_page_config(page_title='NFL Game Prediction', page_icon=':football:')


# Load the trained model
model = load_model('./model-data/model.h5')
scaler = joblib.load('./model-data/scaler.pkl')
encoder = joblib.load('./model-data/encoder.pkl')
categorical_cols = ['team_home', 'team_away', 'stadium']
numerical_cols = ['spread_favorite', 'over_under_line', 'weather_temperature', 'weather_wind_mph', 'weather_humidity', 'schedule_year', 'schedule_month', 'schedule_day_week', 'schedule_day_month' , 'team_home_is_fav']

# Define the input fields
st.title(':football: NFL Game Prediction')

st.header('Input the features for prediction')

format_func = lambda x: f'{x}'

schedule_date = st.date_input('Game Date')
col1, col2 = st.columns(2)
# Assuming the model requires these features
with col1:
    
    team_home = st.selectbox('Team Home', encoder.categories_[0], format_func=format_func)
    team_away = st.selectbox('Team Away', encoder.categories_[1], format_func=format_func , index=1)
    stadium = st.selectbox('Stadium', encoder.categories_[2], format_func=format_func)
    team_home_is_fav = st.checkbox('Is Team Home Favorite?')
    spread_favorite = st.number_input('Spread Favorite', value=-3.0)
with col2:
    over_under_line = st.number_input('Over/Under Line', value=50.0)
    weather_temperature = st.number_input('Weather Temperature', value=70.0)
    weather_wind_mph = st.number_input('Weather Wind MPH', value=6.0)
    weather_humidity = st.number_input('Weather Humidity', value=60.0)


schedule_date = pd.Timestamp(schedule_date)

# Create a DataFrame for the input features
input_features = pd.DataFrame( {
    'team_home': [team_home],
    'team_away': [team_away],
    'stadium': [stadium],
    'team_home_is_fav': [int(team_home_is_fav)] , # very important
    'spread_favorite': [spread_favorite],
    'over_under_line': [over_under_line],
    'weather_temperature': [weather_temperature],
    'weather_wind_mph': [6.29],
    'weather_humidity': [59.44],
   'schedule_year': [schedule_date.year],
    'schedule_month': [schedule_date.month],
    'schedule_day_week': [schedule_date.dayofweek + 1],  # Adjusting for 1=Monday, ..., 7=Sunday
    'schedule_day_month': [schedule_date.day],

})
st.markdown("---")
# Make predictions
if st.button('Predict the score'):
    encoded_cats = encoder.transform(input_features[categorical_cols])

    # Normalize numerical columns

    scaled_nums = scaler.transform(input_features[numerical_cols])

    # Combine processed features
    X_predict = np.hstack([encoded_cats, scaled_nums])
    prediction = model.predict(X_predict)
    
    st.subheader('Prediction using deep learning:')
    col3, col4 = st.columns(2)

    # Determine the winner and loser
    home_score = int(prediction[0][0])
    away_score = int(prediction[0][1])
    home_prob = prediction[0][2] * 100
    away_prob = (1 - prediction[0][2]) * 100

    if home_score > away_score:
        home_color = "green"
        away_color = "red"
    else:
        home_color = "red"
        away_color = "green"

    with col3:
        st.markdown(f'<p style="color:{home_color};">Team Home ({team_home}):</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{home_color};">  Score: {home_score}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{home_color};">  Probability of Winning: {home_prob:.2f}%</p>', unsafe_allow_html=True)

    with col4:
        st.markdown(f'<p style="color:{away_color};">Team Away ({team_away}):</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{away_color};">  Score: {away_score}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{away_color};">  Probability of Winning: {away_prob:.2f}%</p>', unsafe_allow_html=True)