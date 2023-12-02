import streamlit as st
import pandas as pd
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np

# Load data
df_nse = pd.read_csv("Tesla.csv")

# Data preprocessing
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

scaler = MinMaxScaler(feature_range=(0, 1))
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

# Load additional data
df = pd.read_csv("stock_data.csv")

# Streamlit app
st.set_page_config(page_title="Stock Price Analysis Dashboard", layout="wide")

st.title("Stock Price Analysis Dashboard")

tabs = ["Tesla Stock Data", "Facebook Stock Data"]
selected_tab = st.selectbox("Select Stock Data", tabs)

if selected_tab == "Tesla Stock Data":
    st.header("Actual Closing Price")
    fig_actual = go.Figure(data=go.Scatter(x=train.index, y=valid["Close"], mode='markers'))
    fig_actual.update_layout(title='Actual Closing Price Over Time', xaxis={'title': 'Date'}, yaxis={'title': 'Closing Rate'})
    st.plotly_chart(fig_actual)

    st.header("LSTM Predicted Closing Price")
    fig_predicted = go.Figure(data=go.Scatter(x=valid.index, y=valid["Predictions"], mode='markers'))
    fig_predicted.update_layout(title='LSTM Predicted Closing Price Over Time', xaxis={'title': 'Date'}, yaxis={'title': 'Closing Rate'})
    st.plotly_chart(fig_predicted)

    st.header("Closing Price Distribution")
    fig_distribution = go.Figure(data=go.Histogram(x=valid["Close"], nbinsx=20))
    fig_distribution.update_layout(title='Closing Price Distribution', xaxis={'title': 'Closing Rate'}, yaxis={'title': 'Frequency'})
    st.plotly_chart(fig_distribution)

elif selected_tab == "Facebook Stock Data":
    st.header("Facebook Stocks Analysis")

    # Use a unique key for each multiselect widget
    multiselect_key = "select_stocks_" + selected_tab.replace(" ", "_")
    
    selected_stocks = st.multiselect("Select Stocks", ['AAPL', 'FB', 'MSFT'], default=['FB'], key=multiselect_key)
    dropdown = {"AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft"}

    # Sidebar for customization
    st.sidebar.header("Customize Analysis")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["Line Chart", "Candlestick Chart"], key=multiselect_key+"_sidebar")
    show_volume = st.sidebar.checkbox("Show Volume", value=True, key=multiselect_key+"_volume")

    # Main plot
    st.subheader("Stock Prices Over Time")

    fig_stock_prices = go.Figure()

    # Initialize selected_stocks as an empty list if none are selected
    for stock in selected_stocks:
        if plot_type == "Line Chart":
            fig_stock_prices.add_trace(go.Scatter(x=df[df["Stock"] == stock]["Date"],
                                                  y=df[df["Stock"] == stock]["Close"],
                                                  mode='lines',
                                                  opacity=0.7,
                                                  name=f'Close {dropdown[stock]}',
                                                  textposition='bottom center'))
        elif plot_type == "Candlestick Chart":
            fig_stock_prices.add_trace(go.Candlestick(x=df[df["Stock"] == stock]["Date"],
                                                      open=df[df["Stock"] == stock]["Open"],
                                                      high=df[df["Stock"] == stock]["High"],
                                                      low=df[df["Stock"] == stock]["Low"],
                                                      close=df[df["Stock"] == stock]["Close"],
                                                      name=f'Candlestick {dropdown[stock]}'))

    fig_stock_prices.update_layout(title=f"{plot_type} - Stock Prices for {', '.join(str(dropdown[i]) for i in selected_stocks)}",
                                   xaxis={"title": "Date",
                                          'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month',
                                                                               'stepmode': 'backward'},
                                                                              {'count': 6, 'label': '6M', 'step': 'month',
                                                                               'stepmode': 'backward'},
                                                                              {'step': 'all'}])},
                                   'rangeslider': {'visible': True},
                                   'type': 'date'},
                                   yaxis={"title": "Closing Price (USD)"})

    if show_volume:
        st.subheader("Market Volume")
        fig_volume = go.Figure()

        # Initialize selected_stocks as an empty list if none are selected
        for stock in selected_stocks:
            fig_volume.add_trace(go.Scatter(x=df[df["Stock"] == stock]["Date"],
                                            y=df[df["Stock"] == stock]["Volume"],
                                            mode='lines',
                                            opacity=0.7,
                                            name=f'Volume {dropdown[stock]}',
                                            textposition='bottom center'))

        fig_volume.update_layout(title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_stocks)}",
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

    # Highlight Predicted Closing Prices
    st.subheader("Predicted Closing Prices")
    fig_predicted_prices = go.Figure()

    for stock in selected_stocks:
        fig_predicted_prices.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"],
                                                  mode='lines',
                                                  opacity=0.7,
                                                  line=dict(color='orange', width=2),
                                                  name=f'Predicted {dropdown[stock]}'))

    fig_predicted_prices.update_layout(title=f"Predicted Closing Prices - LSTM Model",
                                       xaxis={"title": "Date",
                                              'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month',
                                                                                   'stepmode': 'backward'},
                                                                                  {'count': 6, 'label': '6M', 'step': 'month',
                                                                                   'stepmode': 'backward'},
                                                                                  {'step': 'all'}])},
                                       'rangeslider': {'visible': True},
                                       'type': 'date'},
                                       yaxis={"title": "Predicted Closing Price (USD)"})
    st.plotly_chart(fig_predicted_prices)

