import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import tensorflow
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler(feature_range=(0,1))

df_nse = pd.read_csv("Tesla.csv")

df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]

new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = load_model("saved_lstm_model.h5")

inputs = new_data[len(new_data)-len(valid)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price



df = pd.read_csv("stock_data.csv")

st.title("Stock Price Analysis Dashboard")

tabs = ["Tesla Stock Data", "Facebook Stock Data"]
selected_tab = st.selectbox("Select dataset", tabs)

if selected_tab == "Tesla Stock Data":
    st.header("Actual closing price")
    fig_actual = go.Figure(data=go.Scatter(x=train.index, y=valid["Close"], mode='markers'))
    fig_actual.update_layout(title='Scatter plot', xaxis={'title': 'Date'}, yaxis={'title': 'Closing Rate'})
    st.plotly_chart(fig_actual)

    st.header("LSTM Predicted closing price")
    fig_predicted = go.Figure(data=go.Scatter(x=valid.index, y=valid["Predictions"], mode='markers'))
    fig_predicted.update_layout(title='Scatter plot', xaxis={'title': 'Date'}, yaxis={'title': 'Closing Rate'})
    st.plotly_chart(fig_predicted)

elif selected_tab == "Facebook Stock Data":
    st.header("Facebook Stocks High vs Lows")
    selected_stocks = st.multiselect("Select stocks", [ 'AAPL', 'FB', 'MSFT'], default=['FB'])
    dropdown = {"AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft"}

    fig_highlow = go.Figure()
    for stock in selected_stocks:
        fig_highlow.add_trace(go.Scatter(x=df[df["Stock"] == stock]["Date"],
                                         y=df[df["Stock"] == stock]["High"],
                                         mode='lines',
                                         opacity=0.7,
                                         name=f'High {dropdown[stock]}',
                                         textposition='bottom center'))
    fig_highlow.update_layout(title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_stocks)} Over Time",
                              xaxis={"title": "Date",
                                     'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month',
                                                                          'stepmode': 'backward'},
                                                                         {'count': 6, 'label': '6M', 'step': 'month',
                                                                          'stepmode': 'backward'},
                                                                         {'step': 'all'}])},
                                     'rangeslider': {'visible': True},
                                     'type': 'date'},
                              yaxis={"title": "Price (USD)"})
    st.plotly_chart(fig_highlow)

    st.header("Facebook Market Volume")
    fig_volume = go.Figure()
    for stock in selected_stocks:
        fig_volume.add_trace(go.Scatter(x=df[df["Stock"] == stock]["Date"],
                                        y=df[df["Stock"] == stock]["Volume"],
                                        mode='lines',
                                        opacity=0.7,
                                        name=f'Volume {dropdown[stock]}',
                                        textposition='bottom center'))
    fig_volume.update_layout(title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_stocks)} Over Time",
                             xaxis={"title": "Date",
                                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month',
                                                                         'stepmode': 'backward'},
                                                                        {'count': 6, 'label': '6M', 'step': 'month',
                                                                         'stepmode': 'backward'},
                                                                        {'step': 'all'}])},
                                    'rangeslider': {'visible': True},
                                    'type': 'date'},
                             yaxis={"title": "Transactions Volume"})
    st.plotly_chart(fig_volume)
