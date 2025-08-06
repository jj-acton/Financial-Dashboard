import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sigfig import round as sigfig_round
import yfinance as yf
from datetime import datetime
from pandas.plotting import scatter_matrix
from statsmodels.stats.stattools import durbin_watson, jarque_bera

st.set_page_config(
                page_title="Linear Regression model",
                page_icon="📚",
                layout="wide",
                initial_sidebar_state="auto"
                )


DIVIDER_COLOR = 'blue'
plt.style.use('dark_background')

st.title("Linear Regression Modelling")
st.subheader("This page allows you to perform multivariate linear regression analysis on financial data.", divider=DIVIDER_COLOR)
st.write("Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It is widely used in finance to analyse trends and make predictions based on historical data.")
st.latex(r"""
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_n x_n + \epsilon
""")
st.subheader(" ",)

st.markdown("""
    <div style="text-align: justify;">
    The strategy begins by selecting a primary ETF or index to serve as the target security. In addition to the target security, several related indices or ETFs are used as explanatory variables (signals). These additional indices provide information that may influence the behavior of the target security based on correlations or co-movements in global financial markets.\n
    Historical price data for both the target security and the related indices is collected from Yahoo Finance. The features (signals) are constructed by calculating daily price differences (e.g., Open-to-Open returns), and lagged versions of these differences are also included to capture short-term autocorrelation effects.\n
    The dataset is then split into a training set and a test set. A multivariate linear regression model is fitted to the training data using statsmodels OLS regression (smf.ols). The model predicts the next-day return of the target security based on the previous day\'s returns of the related indices.\n
    After training, the model generates predictions (PredictedY) for both the training and test sets. The model\'s predictive performance is evaluated using metrics such as:
    - Root Mean Square Error (RMSE)\n
    - R-squared (R²)
    To simulate trading decisions, the model\'s predicted returns are converted into trading signals:
    - If PredictedY is positive → take a long position (+1)
    - If PredictedY is negative → take a short position (-1)\n
    The daily profit is then calculated by multiplying the actual return of the target security with the trading signal.
    The same logic is applied to the test set to evaluate out-of-sample performance. Finally, the strategy’s performance is assessed using:
    - Cumulative profit (Wealth curve)
    - Sharpe Ratio — to measure risk-adjusted returns
    - Maximum Drawdown — to evaluate the largest peak-to-trough loss during the trading period\n
    This provides a full statistical evaluation of both predictive power and practical trading viability.
    </div>
""", unsafe_allow_html=True)

st.image("images/Signal-Based Strategy Workflow - visual selection.svg")

st.subheader("Enter a Date Range to run strategy")
date_range = st.date_input(
    "",
    value=(datetime(2020, 1, 1), datetime.now()),
    format='DD-MM-YYYY',
    min_value=datetime(2000, 1, 1),
    max_value=datetime.now()
)

ticker_dict = {
    'ftse100': '^FTSE', 
    'spy': 'SPY',
    'nifty50':'^NSEI',
    'aord': '^AXJO',
    'nikkei': '^N225',
    'hsi': '^HSI',
    'daxi': '^GDAXI',
    'cac40': '^FCHI',
    'sp500': '^GSPC',
    'dji': '^DJI',
    'nasdaq': '^IXIC',
}

# Let user select ETF before downloading any data
indice_choice = st.selectbox('Pick an Index or Security to use for the signal based strategy',ticker_dict)

trading_day_count = {}
yahoo_ticker = ticker_dict[indice_choice]
data = yf.download(yahoo_ticker, start=date_range[0], end=date_range[1])[['Open', 'Close', 'High', 'Low', 'Volume']]#type:ignore
data = pd.DataFrame(data)
trading_day_count = len(data)

st.write(f"Number of trading days within date range: {trading_day_count}")
#Shifts down 1 column as top column is just indice_choice name
data.columns = data.columns.droplevel(1)
st.subheader(f"{indice_choice.upper()} Market Data",divider=DIVIDER_COLOR)

st.dataframe(data)
st.subheader(" ")
st.line_chart(data['Open'],x_label='Date', y_label='Open Price ($)',color= "#fca17d")

indicepanel=pd.DataFrame(index=data.index)


#removes choosen etf/indice to so that new list can be used in OLS to use as comparison data
etf_removed_list = list(ticker_dict.keys())
etf_removed_list.remove(indice_choice)

#gets 1day lag data
indicepanel[f"{indice_choice}"] = data['Open'].shift(-1) - data['Open']
indicepanel[f"{indice_choice}_lag1"] = indicepanel[f"{indice_choice}"].shift(1)

#get all other ticker data
for ticker in etf_removed_list:
    ticker_yahoo = ticker_dict[ticker]
    data_temp = yf.download(ticker_yahoo, start=date_range[0], end=date_range[1])[['Open']]#type:ignore
    indicepanel[f"{ticker}"] = data_temp['Open'] - data_temp['Open'].shift(1)

indicepanel = indicepanel.fillna(method='ffill')#type:ignore
indicepanel = indicepanel.dropna()

#Splits data into test/train data
Train = indicepanel.iloc[0:int(data.shape[0]/2-1), :]
Test = indicepanel.iloc[int(data.shape[0]/2-1):, :]
sm = scatter_matrix(Train, figsize=(10, 10))
st.subheader(f"Correlation of {indice_choice.upper()} with other indices using training data.", divider=DIVIDER_COLOR)
st.pyplot(plt.gcf())
#includes only numeric data e.g removes date
numeric_data = Train.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr()


st.subheader("Correlation Matrix", divider=DIVIDER_COLOR)
st.write("This matrix shows the correlation between the target index and other indices in the dataset. A value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation. Values around 0 suggest no correlation.")
styled_corr = corr_matrix.style.background_gradient(cmap='YlGnBu', vmin=-0.5, vmax=1)

#creates a key : index dict
idx_map = {key: i for i, key in enumerate(ticker_dict)}
etf_index = idx_map.get(indice_choice)
corr_array =  Train.corr()[indice_choice]

#gets second largest key and value as largest key value will be itself and equal to 1
second_largest_key = corr_array.nlargest(2).index[1]
second_largest_num = corr_array.nlargest(2).iloc[1]
st.subheader(f"{indice_choice.upper()} is most correlated with {second_largest_key.upper()} and has a correlation score of : {second_largest_num.round(4)}")
st.write(styled_corr)

#Run OLS
formula = f"{indice_choice}~{' + '.join(etf_removed_list)}"
least_squares = smf.ols(formula=formula, data=Train).fit()
st.subheader("Based on this dataset, we run an ordinary least squares regression on the training split using StatsModels.",divider=DIVIDER_COLOR)
st.write(least_squares.summary())
st.header("Regression Metrics", divider=DIVIDER_COLOR)


col1, col2, col3, col4, col5, col6 = st.columns(6)

# R-squared and Adjusted R-squared
col1.metric("R-squared", f"{least_squares.rsquared:.3f}", help="Explains how much of the variance in the dependent variable is explained by the model.", )
col2.metric("Adj. R-squared", f"{least_squares.rsquared_adj:.3f}", help="Adjusted for the number of predictors in the model.")
# F-statistic and p-value
col3.metric("F-statistic", f"{least_squares.fvalue:.2f}", help="Tests if at least one predictor variable has a non-zero coefficient.")
col4.metric("F-stat p-value", f"{least_squares.f_pvalue:.4f}", help="Low value indicates the model is statistically significant.")
# Durbin-Watson statistic
dw = durbin_watson(least_squares.resid)
col5.metric("Durbin-Watson", f"{dw:.2f}", help="Detects autocorrelation in residuals (ideal: ~2.0).")
#Jarque-Bera
jb_stat, jb_pval, _, _,= jarque_bera(least_squares.resid)
col6.metric("Jarque-Bera", f"{jb_stat:.2f}", help="Tests normality of residuals (lower is better).")



Train[f"Predicted_{indice_choice}"] = least_squares.predict(Train)
Test[f"Predicted_{indice_choice}"] = least_squares.predict(Test)


st.header(f"Predicted vs Actual {indice_choice.upper()}",divider=DIVIDER_COLOR)
st.write("This scatter plot shows how well the model’s predictions match the actual values for the chosen index in the training data. Each point represents one observation, with the actual value on the X-axis and the predicted value on the Y-axis. The red dashed line marks where predictions perfectly match the actual values. Points near this line indicate accurate predictions, while points farther away show larger errors. This helps you quickly see how closely the model fits the data.")
fig, ax = plt.subplots()
ax.scatter(Train[f"{indice_choice}"], Train[f"Predicted_{indice_choice}"], alpha=0.4)
ax.axline((0, 0), slope=1, linestyle="--", color="red") # reference line y = x
plt.xlabel(f"Actual {indice_choice.upper()}")
plt.ylabel(f"Predicted {indice_choice.upper()}")
st.pyplot(fig)

st.header(f"Profit of Signal Based Strategy",divider=DIVIDER_COLOR)
st.write("As the strategy calculates the difference between the next day’s and today’s opening prices to measure the actual return. A positive return means the price increased, and a negative return means it decreased. By shifting this return by one day, the model uses the previous day’s return as a signal. If the lagged return is positive, the strategy sets the Order column to 1, indicating a buy (long) position. If it’s negative or zero, the Order is set to -1, signaling a sell (short) position. This approach turns past price movements into actionable trading decisions.")

#Train data - if value is positive 
Train['Order'] = [1 if sig>0 else -1 for sig in Train[f"Predicted_{indice_choice}"]]
Train['Profit'] = Train[f"{indice_choice}"] * Train['Order']
Train['Wealth'] = Train['Profit'].cumsum()
train_df = pd.DataFrame(Train)

#Test data
Test['Order'] = [1 if sig>0 else -1 for sig in Test[f"Predicted_{indice_choice}"]]
Test['Profit'] = Test[f"{indice_choice}"] * Test['Order']
Test['Wealth'] = Test['Profit'].cumsum()
test_df = pd.DataFrame(Test)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Data Profit Results")
    st.write(train_df[[f"{indice_choice}", f"{indice_choice}_lag1", "Order", "Profit", "Wealth"]])
with col2:
    st.subheader("Test Data Profit Results")
    st.write(test_df[[f"{indice_choice}", f"{indice_choice}_lag1", "Order", "Profit", "Wealth"]])

#wealth plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(10,10), layout="constrained")
fig.suptitle(f"{indice_choice.upper()} Training and Test data vs Buy and Hold Cumulative Profit")
ax1.plot(Train['Wealth'], label = "Signal Based Train Strategy")
ax1.plot(Train[f"{indice_choice}"].cumsum(), label = "Buy and Hold")
ax1.set(xlabel="Date", ylabel="Cumulative Profit ($)")

ax2.plot(Test['Wealth'], label = "Signal Based Test Strategy")
ax2.plot(Test[f"{indice_choice}"].cumsum(), label = "Buy and Hold")
ax2.set(xlabel="Date", ylabel="Cumulative Profit ($)")
ax1.legend()
ax2.legend()
st.pyplot(fig)

st.subheader(f"Total profit made simulating the Training data: ${Train['Profit'].sum().round(2)}")
st.subheader(f"Total profit made Buying and Holding the Training data: ${Train[f'{indice_choice}'].sum().round(2)}")

st.subheader(f"Total profit made simulating the Test data: ${Test['Profit'].sum().round(2)}")
st.subheader(f"Total profit made Buying and Holding the Test data: ${Test[f'{indice_choice}'].sum().round(2)}")

Train['Open'] = data.loc[Train.index, 'Open']
Test['Open'] = data.loc[Test.index, 'Open']

Train['Wealth'] = Train['Wealth'] + Train.loc[Train.index[0], 'Open']
Test['Wealth'] = Test['Wealth'] + Test.loc[Test.index[0], 'Open']

st.header(f"Sharpe Ratio for {indice_choice.upper()}", divider=DIVIDER_COLOR)
st.write('The Sharpe Ratio is commonly used to gauge the performance of an investment by adjusting for its risk. The higher the ratio, the greater the investment return relative to the amount of risk taken, and thus, the better the investment. The formula to calculate the Sharpe ratio is:')
st.latex(r'''
        \text{Sharpe ratio} = \frac{R_p-R_f}{\sigma_p}
        ''')
st.write('Where:')
st.write('$R_p$ = return of portfolio')
st.write('$R_f$ = risk free rate')
st.write('$\sigma_p$ = standard deviation of the portfolio\'s excess return ')#type:ignore

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Training Data")
    Train['Log Return'] = np.log(Train['Wealth']) - np.log(Train['Wealth'].shift(1))
    dailyr = Train['Log Return'].dropna()
    daily_sharpe_ratio = dailyr.mean()/dailyr.std(ddof=1)
    sharpe_ratio = (np.sqrt(trading_day_count/365))*dailyr.mean()/dailyr.std(ddof=1)

    st.write(f"- The Daily Sharpe ratio of {indice_choice.upper()} is: {daily_sharpe_ratio.round(3)}")
    st.write(f"- The Sharpe ratio from the period {date_range[0]} to {date_range[1]} is {sharpe_ratio.round(3)}")#type:ignore

with col2:
    st.subheader('Test Data')
    Test['Log Return'] = np.log(Test['Wealth']) - np.log(Test['Wealth'].shift(1))
    dailyr = Test['Log Return'].dropna()
    daily_sharpe_ratio = dailyr.mean()/dailyr.std(ddof=1)
    sharpe_ratio = (np.sqrt(trading_day_count/365))*dailyr.mean()/dailyr.std(ddof=1)

    st.write(f"- The Daily Sharpe ratio of {indice_choice.upper()} is: {daily_sharpe_ratio.round(3)}")
    st.write(f"- The Sharpe ratio from the period {date_range[0]} to {date_range[1]} is {sharpe_ratio.round(3)}")#type:ignore

st.header(f"Maximum Drawdown for {indice_choice.upper()}", divider=DIVIDER_COLOR)
st.write("A maximum drawdown (MDD) is the maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained. Maximum drawdown is an indicator of downside risk over a specified time period.")
st.write("It is calculated by the following formula:")
st.latex(r'''
        \text{MDD} = \frac{\text{Trough value} - \text{Peak value}}{\text{Peak value}}\times 100\%
        ''')
#Training data
Train['Peak'] = Train['Wealth'].cummax()
Train['Drawdown'] = (Train['Peak'] - Train['Wealth'])/Train['Peak']
st.subheader(f"- Maximum Drawdown in Training data is {Train['Drawdown'].max()*100:.2f}%")

#Test data
Test['Peak'] = Test['Wealth'].cummax()
Test['Drawdown'] = (Test['Peak'] - Test['Wealth'])/Test['Peak']
st.subheader(f"- Maximum Drawdown in Test data is {Test['Drawdown'].max()*100:.2f}%")

st.header("Limitations of Model", divider=DIVIDER_COLOR)
st.subheader("1. No Transaction Costs or Slippage")
st.write("The model does not account for real-world trading costs such as brokerage fees, bid-ask spreads, or slippage. With a high number of daily trades, especially in signal-based strategies, these costs can quickly erode profits. This is particularly important for retail traders who face higher fees and less efficient execution.")
st.subheader("2. Timezone Lag and Market Influence")
st.write("Indices like FTSE (UK) and AORD (Australia) show unusually high simulated profits, largely because they operate in a different timezone and may lag behind major markets like the US. The model can exploit this by using other indices as early signals. While this may boost performance in-sample, it’s not guaranteed to persist out-of-sample.")
st.subheader("3. Overfitting Risk")
st.write("Although train/test splits are used, there is still a risk the model is overfitting to short-term patterns in the data, especially when correlations are strong but unstable over time. Consistent performance should be tested on rolling windows or out-of-time data.")
st.subheader("4. Simplicity of Model")
st.write("The use of a basic linear regression does not capture nonlinear relationships, interactions, or regime changes in markets. It assumes constant relationships between variables, which is unlikely in dynamic financial systems.")
st.subheader("5. No Volume or Macroeconomic Data")
st.write("The model only uses price-based features. It ignores other useful information like trading volume, volatility regimes, interest rates, or macroeconomic indicators that might improve signal quality.")