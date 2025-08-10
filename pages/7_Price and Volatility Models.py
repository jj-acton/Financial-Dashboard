import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn as skl
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


st.set_page_config(page_title="Price and Volatility Modelling",
                page_icon="🏷️",
                initial_sidebar_state="auto",
                layout="wide"
                )

DIVIDER_COLOUR = "blue"

plt.style.use('dark_background')

# Helper Function to make text justified
def justified_text(text: str):
    st.markdown(f"""
        <div style="text-align: justify;">
            {text}
        </div>
    """, unsafe_allow_html=True)

st.title("Price and Volatility Modelling")
st.subheader("In this Page you can see the use of time-series analysis methods to forecast the price and volitility of Gold. For this we use ARIMA, GARCH and BVAR models.", divider=DIVIDER_COLOUR)

st.header("ARIMA Model", divider=DIVIDER_COLOUR)
justified_text("For the following ARIMA model we will be using three years (754 days) of historic close prices for gold. ARIMA or Auto Regression Integrated Moving Averages are a general approach to time series modelling which integrates Auto Regressive modelling and Moving Average modelling, hence the name. The main assumption for a ARIMA model is that it is non-stationary i.e. the values stabilise about the mean.  A stationary series is comparatively easy to predict because you can simply predict that the statistical properties will be about the same in the future as they were in the past. Working with non-stationary data is possible but difficult with an approach like ARIMA.")
st.write("")

ticker = yf.Ticker("GC=F") #Gold
df = ticker.history(period="3y").reset_index() #set index rather than date
df["Date"] = df["Date"].dt.strftime("%d-%m-%y")
df = df[["Date", "Close"]]
st.write(df.tail())
fig,ax = plt.subplots(figsize=(10,6))
df.plot(x="Date",y="Close",ax=ax,title="Gold Closing Price for past 3 years")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price $")
st.pyplot(fig)

justified_text("As there is clearly an upward trend in this data, we can presume that the data is non-stationary as the mean price increases over the time series. To stabilise this we can take the log value for each point.")

df["Log Close"] = np.log(df["Close"])
fig, ax = plt.subplots(figsize = (10,6))
df.plot(x="Date", y="Log Close",ax=ax, title="Log Gold Closing Price for past 3 years")
ax.set_xlabel("Date")
ax.set_ylabel("Log Close Price $")
st.pyplot(fig)

justified_text("Next we will split our data into training data and test data. We are trying to forecast 30 days so that will be the test data and the previous 724 data entries will be used as the training data.")

split_data = (df.index < len(df) - 30)
df_train = df.loc[split_data, "Log Close"].copy()
df_test = df.loc[~split_data, "Log Close"].copy() #~ is a NOT logical operator. as split_data returns True or False, ~ will return the falsey entries.

st.header("Check if data is stationary", divider=DIVIDER_COLOUR)
st.markdown("""
There are multiple ways to check if the data is stationary for an ARIMA model:
1. Viewing the Time Series Plot. The plots above show our data is not stationary as it exhibits a trend. For less obvious cases, more rigorous methods are needed.
2. Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.
3. Augmented Dickey-Fuller (ADF) test.
""")
st.write("---")

st.subheader("ACF and PACF Methods")
st.markdown("The Autocorrelation Function (ACF) measures how a time series is correlated with its own past values across different lags. For example, the correlation at lag $1$ measures the relationship between $y_t$ and $y_{t-1}$, while lag $2$ measures $y_t$ and $y_{t-2}$. However, if $y_t$ and $y_{t-1}$ are correlated, and $y_{t-1}$ and $y_{t-2}$ are also correlated, then $y_t$ and $y_{t-2}$ will appear correlated in the ACF — even if that correlation is only indirect. The Partial Autocorrelation Function (PACF) solves this by measuring the correlation between $y_t$ and $y_{t-k}$ after removing the effects of all intermediate lags from $1$ to $k-1$. For instance, the PACF at lag $2$ controls for $y_{t-1}$, and the PACF at lag $3$ controls for both $y_{t-1}$ and $y_{t-2}$. The PACF at lag $1$ is identical to the ACF at lag $1$ because there are no intermediate lags to remove. Each partial autocorrelation can be interpreted as the last coefficient in an autoregressive model of order $k$, making the PACF a useful tool for identifying the order $p$ in ARIMA models.")

col1, col2 = st.columns(2)

with col1:
    fig_acf, ax_acf = plt.subplots()
    plot_acf(df_train, ax=ax_acf)
    st.pyplot(fig_acf)

with col2:
    fig_pacf, ax_pacf = plt.subplots()
    plot_pacf(df_train, ax=ax_pacf)
    st.pyplot(fig_pacf)
    
justified_text("The ACF plot shows strong positive correlations that decay slowly over time, indicating that past values influence future values for many periods. The PACF plot displays a significant spike only at lag 1, suggesting that correlations at higher lags are mostly indirect. This pattern is typical of a non-stationary random walk, where shocks have a lasting impact and the series does not revert quickly to a stable mean.")
st.write("---")
st.subheader("ADF Method")
st.markdown("The ADF (Augmented Dickey-Fuller) test checks whether a time series has a unit root, indicating non-stationarity. The null hypothesis states that the series is non-stationary. If the $p$-value is greater than 0.05 (at 95\% confidence), we fail to reject the null hypothesis and conclude the data is likely non-stationary.")

adf_test = adfuller(df_train)
st.subheader(f"$p$-value : {adf_test[1]:.3f}")
justified_text("As the value is greater than 0.05 we fail to reject the null hypothesis and conclude the data is non-stationary as expected. Therefore, we must convert the data from non-stationary to stationary.")
st.write("---")

st.header("Transform Data to Stationary : Differencing", divider=DIVIDER_COLOUR)
justified_text("Differencing is a common method to transform a non-stationary time series into a stationary one by subtracting the previous value from the current value. In this project, we use the log return values instead of the raw close prices because close prices often show trends and changing variance over time, making them non-stationary. Log returns stabilize the variance and help remove trends, making the data more suitable for models like ARIMA that assume stationarity.")
st.subheader("")

df_train_diff = df_train.diff().dropna()
fig, ax = plt.subplots(figsize=(10,6))
df_train_diff.plot(x="Date", y=df_train_diff, ax=ax, title="Transforming Non-Stationary Data to Stationary Using Log Returns")
ax.set_xlabel("Index")
ax.set_ylabel("Log Return")
plt.axhline(linestyle = "--")
st.pyplot(fig)


col1, col2 = st.columns(2)

with col1:
    fig_acf_diff, ax_acf = plt.subplots()
    plot_acf(df_train_diff, ax=ax_acf)
    st.pyplot(fig_acf_diff)

with col2:
    fig_pacf_diff, ax_pacf = plt.subplots()
    plot_pacf(df_train_diff, ax=ax_pacf)
    st.pyplot(fig_pacf_diff)

adf_test = adfuller(df_train_diff)
st.subheader(f"$p$-value : {adf_test[1]:.3f}")

st.markdown(
    "In the first image of “Transforming Non-Stationary Data to Stationary Using Log Returns”, we observe a sharp drop in autocorrelation, shown by the reduced variance of the data centered around zero. Both the ACF and PACF drop sharply after taking the first difference, indicating the time series is much less correlated with its past values and is therefore more stationary. Since the ADF test returns a $p$-value of 0, we confidently conclude the differenced series is stationary, so we set ARIMA ($p,1,q$). "
    "Because the autocorrelation is low, a simple ARIMA ($0,1,0$) model might be expected to work well. However, this model failed to converge properly, as shown by the convergence warning and insignificant parameter estimates, suggesting it does not capture the data patterns well. Therefore, we chose the ARIMA ($0,1,1$) model instead, which provides a better fit and more reliable results. "
    "For more information on ARIMA model parameters [click here](https://www.ibm.com/think/topics/arima-model#:~:text=Data%20Scientist-,Introducing%20ARIMA%20models,to%20forecasting%20time%20series%20data)."
)
st.write("---")
st.header("Fitting the ARIMA model",divider=DIVIDER_COLOUR)
justified_text("Calculated using Statsmodels ARIMA model tool to fit the training data.")

model = ARIMA(df_train, order = (0,1,1))
model_fit = model.fit()
st.write(model_fit.summary())

st.header("Residuals Plotting",divider=DIVIDER_COLOUR)
justified_text("To confirm the model is suitable for time series forecasting, we check the residuals to see if they behave like white noise. Residuals resembling white noise indicate the model has captured the underlying data patterns well and left only random noise unexplained.")

residuals = model_fit.resid[1:] #exclude the first value 
fig, ax = plt.subplots(1,2, figsize=(10,6))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(title="Density", kind="kde", ax=ax[1])
st.pyplot(fig)
justified_text("These plots confirm the data is appropriate for modeling, as the residuals resemble white noise, showing no obvious trends or patterns, and their distribution is approximately normal with a mean close to zero.")
st.write("---")

st.subheader("Resiual ACF and PACF")

col1, col2 = st.columns(2)
with col1:
    fig_acf_resid, ax_acf = plt.subplots()
    plot_acf(residuals, ax=ax_acf)
    st.pyplot(fig_acf_diff)

with col2:
    fig_pacf_resid, ax_pacf = plt.subplots()
    plot_pacf(residuals, ax=ax_pacf)
    st.pyplot(fig_pacf_diff)
justified_text("The residual ACF and PACF plots show minimal autocorrelation, indicating the residuals behave very much like white noise.")
st.write("---")

st.header("Forecasting the ARIMA model",divider=DIVIDER_COLOUR)

forecast_test = model_fit.forecast(len(df_test))
forecast_price = np.exp(forecast_test)  # convert back to price from log price
df["Forecast"] = [None] * len(df_train) + list(forecast_price)


fig, ax = plt.subplots()
df[["Close", "Forecast"]].plot(ax=ax)
ax.set_xlabel("Close Price")
ax.set_ylabel("Index")
st.pyplot(fig)
justified_text("The ARIMA model was applied to three years of daily gold close prices to produce a 30-day forecast. The data was log-transformed to stabilise variance and differenced to produce returns, ensuring stationarity as confirmed by the Augmented Dickey-Fuller test. ACF and PACF plots indicated negligible autocorrelation, and ARIMA (0,1,1) was selected based on the lowest AIC score. Model diagnostics showed residuals consistent with white noise and normally distributed around zero, indicating a good fit. The resulting forecast was nearly flat, reflecting the random walk nature of daily gold prices over short horizons and the absence of strong predictive signals in past returns.")

st.header("Evaluating the model",divider=DIVIDER_COLOUR)


mae = mean_absolute_error(df_test, forecast_test)
mape = mean_absolute_percentage_error(df_test, forecast_test)
rmse = np.sqrt(mean_squared_error(df_test, forecast_test))
st.subheader(f"Mean Absolute Error: {mae:.5f}")
st.subheader(f"Mean Absolute Percent Error: {mape:.5f}")
st.subheader(f"Root Mean Squared Error: {rmse:.5f}")

st.header("ARIMA Model Conclusions",divider=DIVIDER_COLOUR)
st.markdown("""
- The ARIMA ($0,1,1$) model was chosen for its better fit and stable convergence compared to simpler alternatives.
- Using log returns transformed the data to be stationary, as confirmed by ACF, PACF, and ADF tests.
- Residuals show no significant autocorrelation and follow a normal distribution, indicating a good model fit.
- The forecasted values remain stable, reflecting the low volatility in historical data.
- More advanced models like GARCH or BVAR could capture volatility clustering and variable interactions better.
- Overall, this ARIMA model provides a solid baseline forecast but may miss sudden market shifts due to its linear nature.
""")