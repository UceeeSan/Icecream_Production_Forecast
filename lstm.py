import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from predict import *


def main():
    st.set_page_config(page_title='Icecream Demand Forecasting')
    st.image('.\Data\icecream.png',width=100)


    # App name
    st.title('Monthly Icecream Order Demand Forecasting')

    # Number of months to forecast
    n = st.number_input('Enter the number of months to forecast', min_value=1)
    
    if st.button('Predict'):
        # Make predictions based on user input
        predictions = predict(n)
        x=[i for i in range(1,n+1)]
        d={'TimeSteps':x,'Predictions':predictions}
        df=pd.DataFrame(d)
        # Predicted sales
        st.subheader('Forecasted values')
        st.dataframe(d)
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(predictions)
        st.pyplot(fig)

    about_button = st.button('About')
    if about_button:
        st.subheader('About')
        st.markdown('''
            This icecream production demand forecasting app uses an LSTM model. A few things you should know before using this app:

            1. The predictions are not always pin point accurate, you can expect a tolerance of 7-8%.

            2. The model is trained on the data available up until 2020.

            3. The predicted value will be displayed in a tabular form .


        ''')

    st.markdown('---')
    st.markdown('Developed by Yusuf Nadim')

if __name__=='__main__':
    main()

    


