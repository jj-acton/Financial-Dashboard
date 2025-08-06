import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import altair as alt
import requests
import xgboost as xgb
import seaborn as sns
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import TimeSeriesSplit
import re

st.set_page_config(
    page_title="Commodities Forecasting",
    layout= 'wide',
    page_icon="🛢️",
    initial_sidebar_state="auto"
)
plt.style.use('dark_background')

# Helper Function to make text justified
def justified_text(text: str):
    st.markdown(f"""
        <div style="text-align: justify;">
            {text}
        </div>
    """, unsafe_allow_html=True)

DIVIDER_COLOUR = 'blue'
NEWS_API_KEY = st.secrets["NewsAPI"]["NEWS_API_KEY"]
colour_palette = sns.color_palette("pastel")

st.title("Commodities Forecasting",)

st.subheader("""
Within this page you can choose a commodity and get realtime market outlook. Additionally, you can read the latest news on the given commodity through the use of NewsAPI. Finally, we look at forecasting the next 30 days of close prices for the choosen commodity using XGBoost machine learning model.
""", divider=f"{DIVIDER_COLOUR}"
)

st.header("Choose a Commodity to view")
option = st.selectbox("",
    (
        "Wheat",
        "Corn",
        "Soybeans",
        "Coffee",
        "Brent Crude",
        "WTI Crude",
        "Natural Gas",
        "Copper",
        "Gold",
        "Silver",
        "Lithium (ETF)",
        "Uranium (ETF)",
    )
)

ticker_dict = {
    "Wheat": {"ticker": "ZW=F", "unit": "USD per bushel"},
    "Corn": {"ticker": "ZC=F", "unit": "USD per bushel"},
    "Soybeans": {"ticker": "ZS=F", "unit": "USD per bushel"},
    "Coffee": {"ticker": "KC=F", "unit": "USD per pound"},
    "Brent Crude": {"ticker": "BZ=F", "unit": "USD per barrel"},
    "WTI Crude": {"ticker": "CL=F", "unit": "USD per barrel"},
    "Natural Gas": {"ticker": "NG=F", "unit": "USD per MMBtu"},
    "Copper": {"ticker": "HG=F", "unit": "USD per pound"},
    "Gold": {"ticker": "GC=F", "unit": "USD per ounce"},
    "Silver": {"ticker": "SI=F", "unit": "USD per ounce"},
    "Lithium (ETF)": {"ticker": "LIT", "unit": "USD (ETF price)"},
    "Uranium (ETF)": {"ticker": "URA", "unit": "USD (ETF price)"},
}
ticker = yf.Ticker(ticker_dict[option]['ticker'])

unit = ticker_dict[option]['unit']

st.write(f"You selected: {option}")
st.header("Market Outlook")
st.subheader(f"Data for {option} ({unit})")

col1, col2, col3, col4, col5 = st.columns(5, gap="large", vertical_alignment="center")

# Fetch historical data for the selected commodity
# Get 7 days just incase theres missing data or weekend 
history = ticker.history(period="7d")
if history.empty or 'Close' not in history.columns:
    st.error("No historical data available for the selected commodity.")
    st.stop()
else:
    #Remove any close prices without data
    history = history.dropna(subset=["Close"])
    most_recent_close = history["Close"].dropna().iloc[-1]
    most_recent_date = history["Close"].dropna().index[-1].date()#type:ignore
    st.write(f"Latest available price: ${most_recent_close:.2f} (from {most_recent_date})")

col1.metric(label=f"Close Price", value=f"${most_recent_close:.2f}",)

#change from previous close
previous_close = history["Close"].dropna().iloc[-2]
change_in_price = most_recent_close - previous_close
col2.metric(label="Price Difference (1 Day)", value=f"${change_in_price:.2f}", delta=f"{change_in_price/most_recent_close:.2%}",)

#YTD change
price_1year_ago = ticker.history(period="1y")['Close'].iloc[0]
yoy_change = most_recent_close - price_1year_ago

#YoY change
col3.metric(label="Price Difference (YoY)", value=f"${yoy_change:.2f}", delta=f"{(most_recent_close - price_1year_ago)/price_1year_ago:.2%}",)

#52 week high
high = ticker.history(period="1y")['Close'].max()
col4.metric(label="52 Week High", value=f"${high:.2f}",)

#52 week low 
low = ticker.history(period=f"1y")['Close'].min()
col5.metric(label="52 Week Low", value=f"${low:.2f}",)

#display 1 year chart
timeframe = st.radio("Select Timeframe", ("5d", "1mo", "3mo", "6mo", "1y", "5y", "YTD", "Max"), horizontal=True, key="timeframe",index=4 )

data = ticker.history(period=f"{timeframe.lower()}")
# Creates index column for dataframe
data.reset_index(inplace=True)

y_min = data["Close"].min()
y_max = data["Close"].max()
padding = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 1
y_scale = alt.Scale(domain=[y_min - padding, y_max + padding])

chart = alt.Chart(data).mark_line(color='#FCA17D').encode(
    x=alt.X("Date:T", title="Date", axis=alt.Axis(labelAngle=-45)),
    y=alt.Y("Close:Q", title=f"Price ({unit})", scale=y_scale)
).properties(
    width=700,
    height=400,
)
st.altair_chart(chart, use_container_width=True)

# Display latest news by querying keywords
st.header(f"Latest News on {option}")
query_keywords = {
    "Wheat": '"Wheat" AND "market" AND "price"',
    "Corn": '"Corn" AND "market" AND "price"',
    "Soybeans": '"Soybeans" AND "market" AND "price"',
    "Coffee": '"Coffee" AND "market" AND "price"',
    "Brent Crude": '"Brent Crude" AND "oil"',
    "WTI Crude": '"WTI Crude" AND "oil"',
    "Natural Gas": '"Natural Gas" AND "energy" AND "price"',
    "Copper": '"Copper" AND "market" AND "price"',
    "Gold": '"Gold" AND "market" AND "price"',
    "Silver": '"Silver" AND "market" AND "price"',
    "Lithium (ETF)": '"Lithium" AND "ETF"',
    "Uranium (ETF)": '"Uranium" AND "ETF"',
}

query = query_keywords[option]

url = (
    f"https://newsapi.org/v2/everything?"
    f"q={query}&"
    "language=en&"
    "sortBy=publishedAt&"
    "searchIn=title,description&"
    f"apiKey={NEWS_API_KEY}"
)

response = requests.get(url)
data = response.json()
articles = data.get("articles")

#prevents text coming out italic or latex styling
def escape_markdown(text):
    # Escape underscores, asterisks, and dollar signs
    text = re.sub(r'(?<!\\)_', r'\_', text)
    text = re.sub(r'(?<!\\)\$', r'\$', text)
    text = re.sub(r'(?<!\\)<', r"", text)
    text = re.sub(r'(?<!\\)>', r"", text)
    return text


if articles:
    #Gets 5 most relevant articles
    for article in articles[:5]:
        safe_subheader = escape_markdown(article["title"])
        st.subheader(safe_subheader)
        safe_content = escape_markdown(article["description"])
        st.markdown(safe_content)
        st.write(f"[Read more]({article['url']})")
        st.write("---")
else:
    st.write("No recent news found for this commodity.")

st.header(f"Forecasting {option} using XGBoost", divider=f"{DIVIDER_COLOUR}")
st.subheader("XGBoost Forecasting")
col1, col2 = st.columns(2)
with col1:
    justified_text(
            "XGBoost (Extreme Gradient Boosting) is a popular and powerful machine learning algorithm used for both classification and regression tasks. It's known for its high accuracy, efficiency, and scalability. At its heart, XGBoost is a gradient boosting algorithm. Gradient boosting is an ensemble learning technique that combines the predictions of multiple weak learners, typically decision trees, to create a strong learner. The key idea is to sequentially add new models to the ensemble, where each new model is trained to correct the errors made by the existing ensemble. XGBoost incorporates regularization techniques to prevent overfitting, which is a common problem in complex models. Overfitting occurs when a model learns the training data too well, including the noise, and performs poorly on unseen data. XGBoost uses two main types of regularization: L1 regularization (Lasso) and L2 regularization (Ridge). XGBoost uses decision trees as its weak learners.")

col2.image("images/Core Principles of XGBoost - visual selection.svg", width=800)


st.subheader("Underlying Concepts of XGBoost",divider=DIVIDER_COLOUR)
tab1, tab2, tab3, tab4 = st.tabs(["Supervised Learning", "Decision Trees", "Ensemble Learning", "Gradient Boosting"])
with tab1:
    st.image("images/How Supervised Machine Learning Works - visual selection.svg", caption="How Supervised Machine Learning Works", use_container_width=True)
with tab2:
    st.image("images/Decision Trees in Machine Learning_ A Comprehensive Guide - visual selection.svg", caption="Decision Trees", use_container_width=True)
with tab3:
    st.image("images/Ensemble Learning_ A Comprehensive Overview - visual selection.svg", caption="Ensemble Learning", use_container_width=True)
with tab4:
    st.image("images/Core Principles of Gradient Boosting - visual selection.svg", caption="Gradient Boosting", use_container_width=True)
st.write("---")

st.subheader("XGBoost Forecasting Model",  divider=f"{DIVIDER_COLOUR}")
justified_text(f"For this example, we use two years of historical data to forecast the next 30 days of closing prices for {option}. The model is trained solely on past closing prices and predicts the next 30 days based on that history. We use a two-year lookback since we're only forecasting a short period of 30 days. However, if you'd like to capture longer-term patterns, you can increase the lookback period using the buttons below. This can help pick up on seasonal or annual trends. To check how well the model performs, we use time series cross-validation, which ensures it generalises well to future data.")

lookback = st.radio("Lookback Period", ("6mo", "1y", "2y", "3y","5y","10y"), horizontal=True, key="lookback", index=2 )
st.write("---")
df = ticker.history(period=f"{lookback}")
df = df[["Close"]].reset_index()

st.line_chart(df.set_index("Date")["Close"],
            use_container_width=True, height=500,
            x_label="Date",
            y_label=f"Close Price of {option} ({unit})",
            color='#FCA17D'
)
st.subheader("Time Series Cross Validation", divider=f"{DIVIDER_COLOUR}")
justified_text("We use TimeSeriesSplit function from scikit-learn library to split our historical data into training and testing sets in a way that uses chronological order to our advantage. Unlike typical machine learning splits that shuffle data randomly, time series data must always predict the future using only the past. TimeSeriesSplit works by creating multiple splits where each training set includes all data up to a certain point in time, and the test set includes the next block of future data. This helps us evaluate how well our model can generalise to unseen future periods without introducing any look-ahead bias. It mirrors how we would use the model in reality — always training on the past to forecast what comes next. For this example, we will use a 5-fold time series cross-validation approach to ensure that our model is robust and can generalise well to unseen data.")
st.write("---")

tss = TimeSeriesSplit(n_splits=5)
fig, axs = plt.subplots(5, 1, figsize = (10,15), sharex=True)
for i, (train_index, validate_index) in enumerate(tss.split(df)):
    train = df.iloc[train_index]
    validate = df.iloc[validate_index]
    axs[i].plot(train["Date"], train["Close"], label="Train")
    axs[i].plot(validate["Date"], validate["Close"], label="Validate")
    axs[i].set_title(f"Fold {i+1}")
    axs[i].set_ylabel(f"Close Price ({unit})")
    axs[i].legend()
    axs[i].axvline(train["Date"].iloc[-1], color='white', linestyle='--')
axs[i].set_xlabel("Date")
st.pyplot(fig)
st.write("---")

st.subheader("Addition of Features",  divider=f"{DIVIDER_COLOUR}")
justified_text("We will create additional features from the date column in the DataFrame. These features will include the day of the week, day of the month, month, quarter, year, and day of the year. This is done to help the model learn patterns in the data that are related to time. Additionally, we will add lag features to the DataFrame. Lag features are previous values of the target variable (in this case daily close price) that can help the model learn patterns in the data that are related to time. For example, the lag feature for 1 day will be the Close Price from the previous day, the lag feature for 2 days will be the Close Price from two days ago, and so on. ")


def create_features(df):
    df = df.copy()
    df["Day of Week"] = df["Date"].dt.dayofweek
    df["Day of Month"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    return df

df = create_features(df)

def add_lags(df):
    df = df.copy()
    df["1 Day Lag"] = df["Close"].shift(1)
    df["2 Day Lag"] = df["Close"].shift(2)
    df["5 Day Lag"] = df["Close"].shift(5)
    df["7 Day Lag"] = df["Close"].shift(7)
    df["14 Day Lag"] = df["Close"].shift(14)
    df["30 Day Lag"] = df["Close"].shift(30)
    return df

df = add_lags(df)

st.subheader("Visualising Features and Target Relationships", divider=f"{DIVIDER_COLOUR}")
justified_text(f"Here we can visualise the distribution of the target variable (Close Price) across different time frames. This will help us understand how the target variable behaves over time and whether or not there are cyclical patterns in the data.")

#Mapping pandas day of week index
map_day_of_week = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
    }

feature_select = st.selectbox("Select a feature to visualise", 
                            options=[
                                "Day of Week",
                                "Day of Month",
                                "Month", "Quarter",
                                "Year",
                                ],
                            key="feature"
                            )

if feature_select == "Day of Week":
    df["Day of Week"] = df["Day of Week"].map(map_day_of_week)

fig, ax =plt.subplots(figsize=(12, 8))
sns.boxplot(x=feature_select, y="Close", data=df, ax=ax, hue=feature_select, palette="Spectral", legend=False)
ax.set_ylabel(f"Close Price ({unit})")
ax.set_xlabel(feature_select)
ax.set_title(f"Distribution of Close Price by {feature_select} for {option}")
st.pyplot(fig)

st.subheader("Creating the XGBoost Regression Model", divider=f"{DIVIDER_COLOUR}")
justified_text(f"We will create an XGBoost regression model to predict the close price of the {option}. The model will be trained on the training set and evaluated on the test set. We will use the features created earlier as input to the model.Below you can tune the hyperparameters of the XGBoost regression model. The hyperparameters are the parameters that control the learning process of the model. The default values are usually a good starting point, but you can experiment with different values to see if you can improve the model's performance.")

with st.form("hyperparameter_form"):
    n_estimators = st.slider("Number of Estimators - Total number of decision trees the model will build.", min_value=100, max_value=2000, value=1000, step=50)
    
    learning_rate = st.slider("Learning Rate - Controls how much each new tree corrects the errors of the previous ones.", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    
    max_depth = st.slider("Max Depth - Maximum depth of each decision tree, controlling how complex each tree can get.", min_value=1, max_value=20, value=6, step=1)
    
    subsample = st.slider("Subsample - Fraction of the training data used to build each tree to reduce overfitting.", min_value=0.1, max_value=1.0, value=0.8, step=0.01)
    
    colsample_bytree = st.slider("Colsample Bytree - Fraction of features randomly selected for each tree to improve model robustness.", min_value=0.1, max_value=1.0, value=0.8, step=0.01)

    submitted = st.form_submit_button("Finished Tuning Hyperparameters")
    
predictions = []
scores = []
# Perform time series cross-validation
tss = TimeSeriesSplit(n_splits=5)
for train_index, validate_index in tss.split(df):
    train = df.iloc[train_index]
    test = df.iloc[validate_index]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['Day of Week', 'Day of Month', 'Month', 'Quarter', 'Year', '1 Day Lag', '2 Day Lag', '5 Day Lag', '7 Day Lag', '14 Day Lag', '30 Day Lag']
    TARGET = 'Close'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    regression_model = xgb.XGBRegressor(n_estimators=n_estimators,
                                        early_stopping_rounds=50,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        subsample=subsample,
                                        colsample_bytree=colsample_bytree,
                                        )
    
    regression_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    
    y_prediction = regression_model.predict(X_test)
    predictions.append(y_prediction)
    score = np.sqrt(mean_squared_error(y_test, y_prediction))
    scores.append(score)

test["Prediction"] = regression_model.predict(X_test)
df = df.merge(test[["Date", "Prediction"]], on="Date", how="left")

st.subheader(f"XGBoost Regression Model Predictions for {option}", divider=DIVIDER_COLOUR)
fig, ax = plt.subplots(figsize=(12,8))
test.plot(x="Date", y="Close", ax=ax, label="Actual - Test Data")
df.plot(x="Date", y="Prediction", ax=ax, label="XGBoost Prediction")
ax.set_xlabel("Date")
ax.set_ylabel(f"Close Price ({unit})")
ax.legend()
st.pyplot(fig)
st.write(f"Mean score across folds: {np.mean(scores):0.4f}")
st.write(f"Individual Fold scores: {scores}")
justified_text("The above chart shows the actual close prices for the test data and the predicted close prices from the XGBoost regression model. The model has been trained on the training data and evaluated on the test data using time series cross-validation. The lower the score, the better the model's performance.")


st.subheader("Feature Importance", divider=DIVIDER_COLOUR)
justified_text("Feature importance is a technique used to understand the impact of each feature on the model's predictions. It helps identify which features are most influential in predicting the target variable. The higher the importance score, the more significant the feature is in making predictions. E.g. a high importance score for 'Month' would indicate that the month of the year has a significant impact on the close price of the commodity.")

feature_importance = pd.DataFrame(data = regression_model.feature_importances_,
            index=FEATURES, columns=["Importance"]).sort_values(by="Importance", ascending=True)

st.bar_chart(feature_importance, use_container_width=True, height=500, color='#FCA17D', horizontal=True)
st.write(f"The above chart shows the feature importance scores for the XGBoost regression model. The features are sorted in ascending order of importance. Here we can see that the {feature_importance['Importance'].idxmax()} has the highest importance score of {feature_importance.max().values[0]:.3f}, indicating that it has the highest impact on the model's predictions.")

st.subheader(f"Predicting 30 Day Close Price for {option}", divider=DIVIDER_COLOUR)
st.write("Now that we have seen the performance of the XGBoost regression model, we can use it to make predictions for the next 30 days.")
st.write("We will retrain the data to inlcude the latest data and then use the model to predict the next 30 days of close prices. The model will use the features created earlier as input to the model.")

# Retrain the model with the latest data
df = create_features(df)

FEATURES = ['Day of Week', 'Day of Month', 'Month', 'Quarter', 'Year', '30 Day Lag']
TARGET = 'Close'
X_all_features = df[FEATURES]
y_all_features = df[TARGET]

regression_model = xgb.XGBRegressor(n_estimators=n_estimators,
                                    early_stopping_rounds=50,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,)
regression_model.fit(
    X_all_features, y_all_features,
    eval_set=[(X_all_features, y_all_features)],
    verbose=False,
)
#future dates is from todays date to 30 days in the future
future_dates = pd.date_range(df["Date"].max(), df["Date"].max() + pd.Timedelta(days=30), freq='D')
future_df = pd.DataFrame({"Date": future_dates})
future_df.set_index("Date", inplace=False)

#Separtes future dates df from past dates df by a bool
future_df["is_future_prediction"] = True
df["is_future_prediction"] = False

#merges all dataframes i.e historical data and predicted data
originaldf_and_future = pd.concat([df, future_df], axis = 0).reset_index(drop=True)

originaldf_and_future = create_features(originaldf_and_future)
originaldf_and_future = add_lags(originaldf_and_future)

future_with_features = originaldf_and_future.query("is_future_prediction == True").copy()
future_with_features["Future Prediction"] = regression_model.predict(future_with_features[FEATURES])

#plot future predictions
fig, ax = plt.subplots(figsize=(12,8))
df.plot(x="Date", y="Close", ax=ax, label="Historical Close Price")
future_with_features.plot(x="Date", y="Future Prediction", ax=ax, label="30 Day Close Price Prediction")
ax.set_title(f"XGBoost Future Predictions for {option} ({unit})")
ax.set_xlabel("Date")
ax.set_ylabel(f"Close Price ({unit})")
ax.legend()
st.pyplot(fig)

#Convert timestamp to DDMMYYYY
date_str = future_with_features["Date"].iloc[-1].date().strftime('%d/%m/%Y')

st.subheader(f"Using the XGBoost model we can see that the model predicts the price of {option} will be ${future_with_features['Future Prediction'].iloc[-1]:.2f} on the {date_str} using a lookback training period of {lookback}.")

st.subheader("Limitations", divider=DIVIDER_COLOUR)
justified_text("While the XGBoost model captures short and medium-term trends using engineered lag features, it is still prone to cumulative bias in sequential predictions and may overfit to recent patterns in certain folds. The model assumes feature stability over time, which may not hold during structural shifts or periods of high volatility. It also does not account for market sentiment, speculation, or external shocks, which can significantly influence price movements.")
justified_text("From testing, I found that commodities like soybeans, wheat, coffee, and corn tend to be predicted more accurately than metals. This is likely due to their cyclical nature and seasonal trends, which models like XGBoost can pick up when using longer lookback periods. On the other hand, metals are more volatile, with sharp and irregular price movements. XGBoost tends to perform worse here, especially when using longer lookback windows, and often underestimates large price spikes. It seems to handle shorter-term trends better in these cases.")