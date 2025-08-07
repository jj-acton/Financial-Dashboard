import os
os.environ["YFINANCE_NO_CACHE"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sigfig import round as sigfig_round
from alpaca.data.historical import StockHistoricalDataClient
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Stock Analysis",
                page_icon="📙",
                initial_sidebar_state="auto",
                layout="wide"
                )

DIVIDER_COLOUR = 'blue'
plt.style.use('dark_background')

API_KEY = st.secrets["alpaca"]["api_key"]
SECRET_KEY = st.secrets["alpaca"]["secret_key"]
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

st.title("Stock Analysis Dashboard: Moving Average Crossover, Risk Metrics & Statistical Testing")
st.subheader("This dashboard allows you to analyse stock data using a simple Moving Average Crossover Strategy. You can input a stock name, visualize its data, and apply the strategy to generate buy and sell signals. Additionally, you can calculate returns, log returns, and assess the probability of stock price changes.", divider=DIVIDER_COLOUR)

st.subheader("Enter Stock Name")
ticker = st.text_input("Enter a stock ticker symbol (e.g., AAPL, AMZN, GOOGL)", "AAPL")

date_range =st.date_input(
    "Select a date range",
    value=(datetime(2020,1,1), datetime.now()),
    format='DD-MM-YYYY',
    min_value=datetime(2000, 1, 1),
    max_value='today'
    )


class MovingAverageStrategy:
    def __init__(self, ticker, date_range):
        self.ticker = ticker.upper() 
        self.date_range = date_range
        self.number_short = 20
        self.number_long = 100
        self.fig = None
        self.ax = None
        self.stock_drop = 0
        self.data = self.fetch_yf_data()
        
    
    def fetch_yf_data(self):
        try:
            self.data = yf.download(self.ticker, start=self.date_range[0], end=self.date_range[1])#type:ignore
            if self.data is None or self.data.empty:
                st.error("Data could not be fetched. Check date range or ticker.")
                self.data = None
                return None
            #Removes stock name as first row and gives index
            self.data.columns = self.data.columns.droplevel(1)
            self.data = self.data.reset_index()
            self.data['Date'] = self.data['Date'].dt.strftime('%d-%m-%Y')
            return self.data
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
            self.data = None
            return None



    def num_ticks(self):
        if self.data is None or self.data.empty:
            st.error(" Num Ticks Data could not be fetched.")
            self.data = None
            return None
        num_days =self.data.shape[0]

        if num_days <= 30:
            return 5
        elif num_days <= 60:
            return 8
        elif num_days <= 120:
            return 12
        elif num_days <= 360:
            return 16
        else:
            return 20
        
            
    def basic_stock_info(self):
        
        if self.data is None or self.data.empty:
            st.warning("No stock data to display.")
        else:
            st.subheader(f"{self.ticker} Stock Data", divider=f'{DIVIDER_COLOUR}')
            if st.checkbox("Show raw data"):
                st.write(self.data)
            st.line_chart(self.data['Close'], x_label='Index', y_label='Close Price ($)', color="#fca17d")
            st.write(f'Number of rows: {self.data.shape[0]}')
            st.write(f'Number of columns: {self.data.shape[1]}')
            st.subheader(f'Summary of {ticker.upper()} Stock Data', divider=f'{DIVIDER_COLOUR}')
            st.write(self.data.loc[:, self.data.columns != 'Date'].describe()
)
            st.write(f"Data fetched for {self.ticker} from {self.date_range[0].strftime('%d-%m-%Y')} to {self.date_range[1].strftime('%d-%m-%Y')}.")
        
    def get_user_input(self):
        st.subheader(f'Moving Average Crossover Strategy on {ticker.upper()} Stock Data', divider=f'{DIVIDER_COLOUR}')
        self.number_short = st.number_input("Insert a number for the short-term moving average", format="%0.0f", value=float(20))
        st.write("The current number is ", self.number_short)
        self.number_long = st.number_input("Insert a number for the long-term moving average", format="%0.0f", value=float(100))
        st.write("The current number is ", self.number_long)
        
    def calculate_moving_averages(self):
        if self.data is None or self.data.empty:
                st.error("Moving Averages data could not be fetched. Check date range or ticker.")
                self.data = None
                return None
        #calculate short and long rolling average
        self.data['Short Term MA'] = self.data['Close'].rolling(int(self.number_short)).mean()
        self.data['Long Term MA'] = self.data['Close'].rolling(int(self.number_long)).mean()

    def plot_moving_averages(self):
        if self.data is None or self.data.empty:
                st.error("Plot Moving Averages data could not be fetched. Check date range or ticker.")
                self.data = None
                return None
        fig, ax = plt.subplots()
        ax.plot(self.data['Date'], self.data['Close'], label='Close Price')
        ax.plot(self.data['Date'], self.data['Short Term MA'], label=f'{int(self.number_short)}-Day Moving Average', color='orange')
        ax.plot(self.data['Date'], self.data['Long Term MA'], label=f'{int(self.number_long)}-Day Moving Average', color='red')
        ax.set_xticks(self.data['Date'][::120])  # Show every 30th date
        ax.set_xticklabels(self.data['Date'][::120], rotation=30, fontsize = 8)
        ax.set(xlabel='Date', ylabel='Price ($)')
        ax.legend()
        st.pyplot(fig)

    def generate_signals(self):
        if self.data is None or self.data.empty:
                st.error("Generate Signals data could not be fetched.")
                self.data = None
                return None
        self.data['Signal'] = 0
        self.data.loc[self.data['Short Term MA'] > self.data['Long Term MA'], 'Signal'] = 1  # Long
        self.data.loc[self.data['Short Term MA'] < self.data['Long Term MA'], 'Signal'] = -1  # Short
        self.data['Crossover'] = self.data['Signal'].diff()
        crossover_dates = self.data[self.data['Crossover'].abs() == 2][['Date', 'Signal', 'Crossover']]
        return crossover_dates

    def display_signals(self, crossover_dates):
        st.subheader('Moving Average Crossover Signals')
        col1, col2 = st.columns(2)
        for date in crossover_dates['Date']:
            row = crossover_dates[crossover_dates['Date'] == date].iloc[0]
            if row['Crossover'] == 2:
                col1.markdown(f"**Buy** on **{date}** (signal={row['Signal']})")
            elif row['Crossover'] == -2:
                col2.markdown(f"**Sell** on **{date}** (signal={row['Signal']})")

    def show_explanation(self):
        col1, col2, col3 = st.columns(3)
        if col2.button('Show Explanation'):
            st.markdown("""
            The Moving Average Crossover Strategy is a popular trading strategy that uses two moving averages to identify potential buy and sell signals. 
            - When the short-term moving average crosses above the long-term moving average, it indicates upward momentum, suggesting a buy signal.
            - Conversely, when the short-term moving average crosses below the long-term moving average, it indicates downward momentum, suggesting a sell signal.
            -  It does not predict future direction but shows trends.
            """)


    def show_returns(self):
        if self.data is None or self.data.empty:
                st.error("Show Returns data could not be fetched.")
                self.data = None
                return None
        # Calculate Daily Close Price Differece
        self.data['PriceDiff'] = self.data['Close'].shift(-1) - self.data['Close'] 

        #Calculate Daily Return
        self.data['Return'] = self.data['PriceDiff'] / self.data['Close'] 

        #Calculate whether daily stock is winner or loser
        self.data['Direction'] = [1 if self.data['PriceDiff'].iloc[i] > 0 else -1 for i in range(len(self.data))] 
        st.write(f'Number of winning days: {self.data["Direction"].sum()} i.e close price increased from previous day')
        st.write(f'Number of losing days: {len(self.data) - self.data["Direction"].sum()} i.e close price decreased from previous day')

    def price_diff_plot(self):
        if self.data is None or self.data.empty:
                st.error(" Price Diff Plot Data could not be fetched.")
                self.data = None
                return None
        st.header('Profit Analysis', divider=f'{DIVIDER_COLOUR}')
        st.write(f"Here we can see the daily price change in {self.ticker}. An increased change in price is a sign of a period of volatilty.")
        fig, ax = plt.subplots()
        ax.plot(self.data['Date'], self.data['PriceDiff'], label='Daily Price Difference')
        ax.set_xticks(self.data['Date'][::120])  # Show every 30th date
        ax.set_xticklabels(self.data['Date'][::120], rotation=30, fontsize = 8)
        ax.set(xlabel='Date', ylabel='Price Difference ($)', title=f'Daily Price Difference of {ticker.upper()} Stock')
        plt.axhline(y=0, color = "red", linestyle='--' )
        st.pyplot(fig)
        st.write(" ")
        st.write(self.data.reset_index(drop=True))
    
    def profit_plot(self):
        if self.data is None or self.data.empty:
                st.error("Profit Plot Data could not be fetched.")
                self.data = None
                return None
        self.data['Shares'] = [1 if self.data.loc[ei, 'Short Term MA'] > self.data.loc[ei, 'Long Term MA'] else 0 for ei in self.data.index] #type:ignore
        self.data['Close1'] = self.data['Close'].shift(-1)
        self.data['Profit'] = [self.data.loc[ei, 'Close1'] - self.data.loc[ei, 'Close'] if self.data.loc[ei, 'Shares']==1 else 0 for ei in self.data.index] #type:ignore
        
        fig, ax = plt.subplots()
        ax.plot(self.data['Date'], self.data['Profit'], label='Daily Profit')
        ax.set_xticks(self.data['Date'][::120])
        ax.set_xticklabels(self.data['Date'][::120], rotation=30, fontsize = 8)
        ax.set(xlabel='Date', ylabel='Profit ($)', title='Daily Profit from Moving Average Strategy')
        plt.axhline(y=0, color='red', linestyle = "--")
        st.pyplot(fig)
    
    def cumsum_profit(self):
        if self.data is None or self.data.empty:
            st.error("Cumsum Profit Data could not be fetched.")
            self.data = None
            return None
        self.data['Cumulative Profit'] = self.data['Profit'].cumsum()
        fig, ax = plt.subplots()
        ax.plot(self.data['Date'], self.data['Cumulative Profit'], label='Cumulative Profit')
        ax.legend()
        ax.set_xticks(self.data['Date'][::120])
        ax.set_xticklabels(self.data['Date'][::120], rotation=30, fontsize = 8)
        ax.set(xlabel='Date', ylabel='Cumulative Profit ($)', title=f'Total profit = ${self.data["Cumulative Profit"].iloc[-2].round(2)}')
        st.pyplot(fig)
        
    
    def log_returns(self):
        if self.data is None or self.data.empty:
            st.error("Log Returns Data could not be fetched.")
            self.data = None
            return None
        st.subheader(f'Log Returns for {ticker.upper()} Stock', divider=f'{DIVIDER_COLOUR}',)
        st.write('Logarithmic returns are often more useful than regular arithmetic returns in financial analysis because they are time-additive. This means the log return over a longer period can be calculated simply by summing the log returns of shorter intervals. This makes them especially convenient when analyzing returns over time. Additionally, log returns tend to be more symmetrically distributed and closer to a normal distribution, which is beneficial for statistical modeling and risk management. They also handle small percentage changes more accurately and are consistent across different asset prices, making comparisons more straightforward. Many financial models, such as the Black-Scholes option pricing model, are built using log returns due to their mathematical properties.')
        
        #Log Return Calculationm
        self.data['LogReturn'] = np.log(self.data['Close'].shift(-1) / self.data['Close'])
        # Remove unneccesary particular columns
        st.write(self.data.reset_index(drop=True).drop(columns=['Short Term MA','Long Term MA','Signal','Crossover','Direction','Shares','Close1','Profit', 'Cumulative Profit'], axis=1,).head(11))
        
        self.mu = self.data['LogReturn'].mean()
        self.sigma = self.data['LogReturn'].std(ddof = 1)
        
        self.density = pd.DataFrame()
        self.density['x'] = np.arange(self.data['LogReturn'].min()-0.01, self.data['LogReturn'].max()+0.01, 0.001)
        self.density['pdf'] = norm.pdf(self.density['x'], self.mu, self.sigma)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.data['LogReturn'], linestyle='-', markersize=2, label='Log Returns')
        ax.axhline(y=0, color='red', linestyle = "--")
        ax.set_title(label = f"Log Returns of {ticker.upper()} Stock", fontsize=20)
        ax.set_ylabel(ylabel="Log Return", fontsize=20)
        ax.set_xlabel(xlabel= "Index", fontsize= 20)
        st.pyplot(fig)
        st.write(f'The mean (μ) of the log returns is approximately {sigfig_round(self.mu, sigfigs=4)} and the standard deviation (σ) is approximately {sigfig_round(self.sigma, sigfigs=4)}.')
    
        fig, ax = plt.subplots()
        # Plot histogram directly
        ax.hist(self.data['LogReturn'], bins=50, color='purple', alpha=0.8, label="Histogram")
        # Plot the PDF
        ax.plot(self.density['x'], self.density['pdf'], color='orange', label='PDF')
        ax.set(xlabel='Log Return', ylabel='Density', title=f'Log Returns of {ticker.upper()} Stock Histogram and PDF')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        st.write(f'Here we use the properties of the normal distribution to calculate the mean (μ) and standard deviation (σ) of the log returns. The mean represents the average daily return, while the standard deviation indicates the volatility or risk associated with the stock.' )
        
        
    def stock_change_percentage(self):
        st.subheader(f'Calculating the probability the stock price will drop over a certain percentage in a day', divider=f'{DIVIDER_COLOUR}')
        
        st.write(f'To calculate the probability of {ticker.upper()} stock dropping more than a certain percentage in a day, we can use the properties of the normal distribution. In particular, we can use the cumulative distribution function (CDF) of the normal distribution to find the probability of a stock dropping more than a certain percentage from its mean return.')
        
        self.daily_stock_drop = st.number_input(f"Insert a value to see the probability of {ticker.upper()} dropping (%) in a day.", min_value=0.0, max_value=100.0, value=1.0, format="%0.2f")
        
        stock_drop_percentage = norm.cdf(-self.daily_stock_drop/100, self.mu, self.sigma)
        st.write(f'The probability of {ticker.upper()} stock dropping more than {float(self.daily_stock_drop):.2f}% in a day is approximately {sigfig_round(stock_drop_percentage*100, sigfigs=4)} %.')
    
        
        self.yearly_stock_drop = st.number_input(f"Insert a value to see the probability of {ticker.upper()} dropping (%) in a trading year.", min_value=0.0, max_value=100.0, value=5.0, format="%0.2f")
        
        self.mu220 = self.mu * 220  # Assuming 220 trading days in a year
        self.sigma220 = self.sigma * np.sqrt(220)
        
        yearly_stock_drop_percentage = norm.cdf(-self.yearly_stock_drop/100, self.mu220, self.sigma220)
        st.write(f'The probability of {ticker.upper()} stock dropping more than {float(self.yearly_stock_drop)}% in a trading year is approximately {sigfig_round(yearly_stock_drop_percentage*100, sigfigs=4)} %.')
        st.divider()
    
    def value_at_risk(self):
        st.header(f'Calculating Value at Risk (VaR) for {ticker.upper()} Stock', divider=f'{DIVIDER_COLOUR}')
        st.write('Value at Risk (VaR) is a statistical measure used to assess the risk of loss on an investment. It estimates how much a set of investments might lose, given normal market conditions, in a set time period such as one day or one year, at a given confidence level.')
        st.write('The confidence level is the probability that the loss will not exceed a certain amount. For example, a 95% confidence level means that there is a 5% chance that the loss will exceed the VaR amount.')
        st.write('A higher confidence level means a lower VaR, as it accounts for more extreme losses and gives a higher certainty that the true mean is within the given range. Conversely, a lower confidence level results in a higher VaR, as it allows for more extreme losses to be included in the calculation. ')
        confidence_level = st.number_input("Insert a confidence level (0-99) for VaR calculation", min_value=0.00, max_value=99.00, value=95.00, format="%0.2f")
        alpha = 1 - confidence_level / 100
        
        # Calculate VaR
        var = norm.ppf(alpha, self.mu, self.sigma)
        st.write(f'The Value at Risk (VaR) at {confidence_level}% confidence level is approximately {abs(var)*100:.2f}%.')
        st.write(f'This means there is a {confidence_level}% chance that the stock will not drop more than {abs(var)*100:.2f}% in a day.')
    
    def average_return_with_CI(self):
        if self.data is None or self.data.empty:
            st.error("Avereage Return with CI Data could not be fetched.")
            self.data = None
            return None
        st.subheader(f'Calculating Average Return with Confidence Interval for {ticker.upper()} Stock', divider=f'{DIVIDER_COLOUR}')
        st.write('The average return is the mean of the daily returns, and the confidence interval provides a range within which we can expect the true mean return to lie with a certain level of confidence.')
        
        confidence_interval = st.number_input("Insert a confidence level (0-100 ) for the confidence interval", min_value=0.00, max_value=100.00, value=95.00, key='confidence_level', format="%0.2f")
        
        self.sample_size= self.data['LogReturn'].shape[0]
        self.sample_mean = self.data['LogReturn'].mean()
        self.sample_std = self.data['LogReturn'].std(ddof=1)
        alpha = 1 - confidence_interval / 100
        z_left = norm.ppf(alpha / 2)
        z_right = -z_left
        
        lower_bound = self.sample_mean + z_left * self.sample_std
        upper_bound = self.sample_mean + z_right * self.sample_std
        
        st.write(f'A {confidence_interval}% confidence interval tells you that there will be {confidence_interval}% chance that the average daily stock return lies between {sigfig_round(lower_bound*100, sigfigs=4)}% and {sigfig_round(upper_bound*100, sigfigs=4)}%.')
        
    def hypothesis_testing(self):
        st.subheader(f'One Tailed Hypothesis Testing for {ticker.upper()} Stock', divider=f'{DIVIDER_COLOUR}')
        st.write('Hypothesis testing is a statistical method used to make inferences about a population based on a sample that can be used to confirm a financial claim or theory.')
        
        st.write('In this case, we will test the hypothesis that the average daily return of the stock is greater than zero.')
        
        st.write('Hypothesis testing contains four steps; define the hypothesis, set the criteria, calculate the statistic, and reach a conclusion.')
        
        st.markdown('1. **Define the hypothesis**: The null hypothesis ($H_0$) is that the average daily return of the stock is less than or equal to zero, and the alternative hypothesis ($H_1$) is that the average daily return of the stock is greater than zero.')
        st.latex(r'H_0: \mu_0 \leq 0 \\ H_1: \mu_0 > 0')
        st.write('if the sample size $n$ is large enough, we can use $z$-distribution, instead of $t$-distribtuion, to calculate the test statistic.')
        
        alpha = st.number_input('Insert a significance level (alpha) for the hypothesis test', min_value=0.00, max_value=1.00, value=0.05, step=0.01, format="%0.2f", key='alpha')
        
        st.write(f'2. **Set the criteria**: We will use a significance level of {alpha}, which means that we will reject the null hypothesis if the $z$-value less than the test statistic.')
        
        z_right = norm.ppf(1 - alpha)  # Critical value for one-tailed test
        st.write(f'The critical value for a one-tailed test at a significance level of alpha is approximately {sigfig_round(z_right, sigfigs=6)}.')
        
        st.write('3. **Calculate the test statistic**: The test statistic is calculated as follows:')
        st.latex(r'{Z} = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}')
        
        zhat = (self.sample_mean-0)/(self.sample_std/self.sample_size**0.5)
        
        st.write(f'The test statistic ($Z$) is approximately {sigfig_round(zhat, sigfigs=6)}.')
        
        st.write('4. **Reach a conclusion**: If the test statistic is greater than the critical value, we reject the null hypothesis and conclude that the average daily return of the stock is greater than zero.')
        if zhat > z_right:
            st.write(f'The test statistic ($Z$) {sigfig_round(zhat, sigfigs=6)} is greater than the critical value ($z$) {sigfig_round(z_right, sigfigs=6)}, so we reject the null hypothesis and conclude that the average daily return of the stock is greater than zero.')
        else:
            st.write(f'The test statistic ($Z$) {sigfig_round(zhat, sigfigs=6)} is less than the critical value ($z$) {sigfig_round(z_right, sigfigs=6)}, so we fail to reject the null hypothesis and conclude that there is not enough evidence to support the claim that the average daily return of the stock is greater than zero.')
    
        st.subheader('Conclusion', divider=f'{DIVIDER_COLOUR}')
        st.write(f'This dashboard provides a comprehensive analysis of the {ticker.upper()} stock using the simple Moving Average Crossover Strategy. It allows users to visualise stock data, calculate moving averages, generate buy and sell signals, and assess the stock\'s performance through various metrics such as returns, log returns, Value at Risk (VaR), and hypothesis testing. The dashboard also provides insights into the stock\'s risk and return characteristics.')
        st.subheader('Limitations of The Moving Crossover Strategy', divider=f'{DIVIDER_COLOUR}')
        st.markdown("""
        - The Moving Average Crossover Strategy is a simple strategy that may not be effective in all market conditions. It is based on historical data and may not accurately predict future price movements.
        - Moving averages are lagging indicators. They follow price action, not predict it. A crossover tells you what has happened, not what will happen. This often means you enter trades late and exit late — missing tops and bottoms.
        - It’s easy to overfit moving average lengths to historical data. What looks great on past data often doesn’t hold up in future conditions.
        - Moving averages don’t know why prices move. They ignore macro events, earnings, supply shocks, or changing volatility regimes. This makes them very basic compared to more sophisticated strategies.
        - The dashboard relies on the availability of stock data from Yahoo Finance, which may not always be accurate or up-to-date.
        - The dashboard does not account for transaction costs, slippage, or other real-world trading factors that can affect the performance of the Moving Average Crossover Strategy.
    
        """)
        
        
    
    def run(self):
        self.basic_stock_info()
        self.get_user_input()
        self.calculate_moving_averages()
        self.plot_moving_averages()
        crossover_dates = self.generate_signals()
        self.display_signals(crossover_dates)
        self.show_explanation()
        self.show_returns()
        self.price_diff_plot()
        self.profit_plot()
        self.cumsum_profit()
        self.log_returns()
        self.stock_change_percentage()
        self.value_at_risk()
        self.average_return_with_CI()
        self.hypothesis_testing()
        
            
strategy = MovingAverageStrategy(ticker, date_range)
strategy.run()
if __name__ == "__main__":
     st.write("---")



