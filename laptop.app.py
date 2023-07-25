import streamlit as st
import pickle
import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # or import the specific model you are using

# The rest of your code here...

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())
laptop_type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight = st.number_input('Weight of the Laptop')

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size')

cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Create a dictionary with the user input
    user_input = {
        'Company': company,
        'TypeName': laptop_type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'IPS': ips,
        'ScreenSize': screen_size,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }

    # Convert the user input into a dataframe
    input_df = pd.DataFrame([user_input])

    # Use the loaded model to predict the laptop price
    predicted_price = pipe.predict(input_df)

    # Display the predicted price to the user
    st.write(f"Predicted Price: ${predicted_price[0]:.2f}")
