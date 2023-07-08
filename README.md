# Stock Price Prediction and Visualization

This project aims to predict stock prices using an LSTM (Long Short-Term Memory) model and visualize the predictions through interactive dashboards. It utilizes machine learning techniques to analyze historical stock data and generate future price predictions.

## Model 1: LSTM Stock Price Prediction

The first part of the project focuses on training an LSTM model to predict stock prices. It uses historical stock data and leverages the Keras library to build and train the model. The model is trained on a dataset of historical stock prices and can be used to forecast future price trends. The model is saved and later utilized in the visualization dashboard.

## Dashboard: Stock Price Analysis and Prediction

The second part of the project involves creating an interactive dashboard using Streamlit. This dashboard provides users with a user-friendly interface to explore stock price data and obtain predictions based on the trained LSTM model. The dashboard allows users to select different stocks, view historical price trends, compare actual and predicted closing prices, and analyze market volume over time. It leverages Plotly graphs to visualize the data and provides an intuitive way to interact with the stock price analysis.

## Dependencies

To run the Streamlit dashboard, ensure that the following dependencies are installed:

- Streamlit: `pip install streamlit`
- Pandas: `pip install pandas`
- Numpy:  `pip install numpy`
- Plotly: `pip install plotly`
- Keras: `pip install keras`
- scikit-learn: `pip install scikit-learn`

Once the dependencies are installed, start the Streamlit development server by running the following command:
`streamlit run dashboard.py`

This will launch the Streamlit app, and you can access the interactive dashboard through your web browser. Select the desired dataset and explore the stock price analysis and predictions.

Feel free to customize the code, modify the visualizations, or incorporate additional features to meet your specific requirements.

For any questions or issues, please Email [mwassdennoh7@gmail.com].


