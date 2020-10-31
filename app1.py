import streamlit as st
import pandas as pd 
import numpy as np 
from fbprophet import Prophet 
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation 
from fbprophet.plot import plot_cross_validation_metric
import base64

st.title("Prediction model")
st.set_option("deprecation.showfileUploaderEncoding", False)
# Importing data
df = st.file_uploader('file uploading point', type='csv', encoding='auto')
if df is not None:
	data = pd.read_csv(df)
	data['ds']=pd.to_datetime(data['ds'],errors='coerce')
	st.write(data)
	max_date=data['ds'].max()
	st.write(max_date)

# Select forecast horizon 

periods_input= st.sidebar.number_input('How many periods would you like to forecast into the future?', max_value = 365, min_value = 1)
if df is not None:
	m=Prophet()
	m.fit(data)
# Visualising forecast data 
if df is not None:
	future =m.make_future_dataframe(periods=periods_input)
	forecast=m.predict(future)
	fcst=forecast[['ds' , 'yhat' , 'yhat_lower' , 'yhat_upper']]
	fcst_filtered = fcst[fcst['ds'] > max_date]
	st.write(fcst_filtered)
	fig1=m.plot(forecast)
	st.write(fig1)
	fig2 = m.plot_components(forecast)
	st.write(fig2)


