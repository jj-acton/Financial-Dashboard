import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

st.set_page_config(
                page_title="Black Scholes Model",
                page_icon=":chart_with_upwards_trend:",
                layout="wide",
                initial_sidebar_state="auto"
                )

DIVIDER_COLOUR = "blue"

def justified_text(text: str):
    st.markdown(f"""
        <div style="text-align: justify;">
            {text}
        </div>
    """, unsafe_allow_html=True)

st.title("Options Pricing Calculator")
st.subheader("This app allows you to calculate option prices and Greeks using the Black Scholes Model. You can input the parameters for the model and see the results in real-time.", divider=DIVIDER_COLOUR)
col1, col2, col3= st.columns(3)
with col1:
    st.markdown("""
                <div style="text-align: justify;">
                The Black Scholes Model is a mathematical model used to calculate the theoretical price of an option. An option is a financial derivative that gives the holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined price (the strike price) before or at a specified expiration date. It is widely used in the financial industry for pricing European-style options.
                </div>
                """, unsafe_allow_html=True)
with col2:
    st.markdown("""
                <div style="text-align: justify;">
                The model takes into account several factors, including the current price of the underlying asset, the strike price of the option, the time to expiration, the risk-free interest rate, and the volatility of the underlying asset. The Black Scholes Model assumes that the underlying asset follows a geometric Brownian motion and that markets are efficient.
                </div>
                """, unsafe_allow_html=True)
with col3:
    col1, col2, col3= st.columns(3)
    with st.popover("European Option"):
        st.write("A European option can only be exercised at expiration. The Black-Scholes method is primarily used for pricing European options.", )
    with st.popover("American Option"):
        st.write("An American option can be exercised at any time before expiration. The Black-Scholes model does not apply directly to American options, but it can be used as a starting point for more complex models that account for early exercise features.")
    with st.popover("Asian Option"):
        st.write("For Asian options the payout is depends on the average price of the underlying asset over a certain period, rather than just the price at expiration. The Black-Scholes model can be adapted for Asian options, but it requires additional calculations to account for the averaging process.")

st.header("Black Scholes formula", divider=DIVIDER_COLOUR)

st.subheader("The call and put option price for the Black Scholes Model is defined by:")
col1, col2 = st.columns(2)
with col1:
    st.write("Call Option Price ($C$):")
    st.latex(r"""
        C = S_0 N(d_1) - K e^{-rT} N(d_2)
    """)
with col2:
    st.write("Put Option Price ($P$):")
    st.latex(r"""
        P = K e^{-rT} N(-d_2) - S_0 N(-d_1)
    """)
st.write("Where:")
col1, col2 = st.columns(2)
with col1:
    st.latex(r"""
        d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}
    """)
with col2:
    st.latex(r"""
        d_2 = d_1 - \sigma\sqrt{T}
    """)
st.write(" ")

#Define Variables
with st.form("Black_Scholes_form"):
    S_0 = st.number_input("Current Price of Underlying Asset ($S_0$)", min_value=0.0, value=100.0, step=1.0)
    K = st.number_input("Strike Price ($K$)", min_value=0.0, value=110.0, step=1.0)
    T = st.number_input("Time to Expiration in Years ($T$)", min_value=0.0, value=1.0, step=0.01)
    r = st.number_input("Risk-Free Interest Rate ($r$)", min_value=0.0, value=0.05, step=0.01)
    sigma_stationary = st.number_input("Volatility of Underlying Asset ($σ$)", min_value=0.0, value=0.2, step=0.01)
    submitted = st.form_submit_button("Finished Editing Parameters")

#calculate greeks and call/put 
def calculate_prices_and_greeks(S_0, K, T, r, sigma_stationary):
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma_stationary ** 2) * T) / (sigma_stationary * np.sqrt(T))
    d2 = d1 - sigma_stationary * np.sqrt(T)
    return {
        "call_price": S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        "put_price": K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * norm.cdf(-d1),
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S_0 * sigma_stationary * np.sqrt(T)),
        "vega": S_0 * norm.pdf(d1) * np.sqrt(T),
        "theta": (-S_0 * norm.pdf(d1) * sigma_stationary / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365,
        "rho": K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    }
prices_and_greeks = calculate_prices_and_greeks(S_0, K, T, r, sigma_stationary)

call_price = prices_and_greeks["call_price"]
put_price = prices_and_greeks["put_price"]
delta = prices_and_greeks["delta"]
gamma = prices_and_greeks["gamma"]
vega = prices_and_greeks["vega"]
theta = prices_and_greeks["theta"]
rho = prices_and_greeks["rho"] 

st.write(" ")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div style="background-color:#90ee90; padding:5px; border-radius:20px; text-align:center">
            <p style="color:#1f1f1f; font-size:16px; margin-bottom:5px;">CALL Price</p>
            <p style="color:#1f1f1f; font-size:28px; font-weight:bold;">${call_price:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color:#f4cccc; padding:5px; border-radius:20px; text-align:center">
            <p style="color:#1f1f1f; font-size:16px; margin-bottom:5px;">PUT Price</p>
            <p style="color:#1f1f1f; font-size:28px; font-weight:bold;">${put_price:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.write("---")
# Display Greeks
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Delta", f"{delta:.3f}", border=True)
    st.markdown("""
        <div style="text-align: justify; font-size: 12px;">
        Delta (Δ) is a measure of the sensitivity of an option’s price changes relative to the changes in the underlying asset’s price. 
        </div>
        """, unsafe_allow_html=True)
with col2:
    st.metric("Gamma", f"{gamma:.3f}", border=True)
    st.markdown("""
        <div style="text-align: justify; font-size: 12px;">
        Gamma (Γ) is a measure of the delta’s change relative to the changes in the price of the underlying asset. 
        </div>
        """, unsafe_allow_html=True)
with col3:
    st.metric("Vega", f"{vega:.3f}", border=True)
    st.markdown("""
        <div style="text-align: justify; font-size: 12px;">
        Vega (ν) is a measure of the sensitivity of an option’s price to changes in the volatility of the underlying asset.
        </div>
        """, unsafe_allow_html=True)
with col4:
    st.metric("Theta", f"{theta:.3f}", border=True)
    st.markdown("""
    <div style="text-align: justify; font-size: 12px;">
    Theta (θ) is a measure of the sensitivity of the option price relative to the option’s time to maturity.
    </div>
        """, unsafe_allow_html=True)
    
with col5:
    st.metric("Rho", f"{rho:.3f}", border=True)
    st.markdown("""
    <div style="text-align: justify; font-size: 12px;">
    Rho (ρ) measures the sensitivity of the option price relative to interest rates. 
    </div>
        """, unsafe_allow_html=True)

st.write(" ")
st.header("Options Price - Interactive Heatmap", divider=DIVIDER_COLOUR)
st.write("This heatmap explores the relationship between the two most sensitive parameters of the Black Scholes Model, the volatility ($σ$) of the underlying asset and the spot price ($S_O$).")

with st.form("Interactive_Heatmap_form"):
    sigma = st.slider("Volatility ($σ$) Range ", value=[0.05,0.5], min_value=0.0, max_value=1.0, step=0.01)
    spot_price_min = st.number_input("Minimum Spot Price ($)", min_value=0.0, value=50.0, step=1.0)
    spot_price_max = st.number_input("Maximum Spot Price ($)", min_value=0.0, value=150.0, step=1.0)
    submitted = st.form_submit_button("Finished Editing Parameters")

st.header("Call/Put Option Premiums and PnL at Expiry",divider=DIVIDER_COLOUR)
st.write(f"Current Price of Underlying : ${S_0}")
st.write(f"Strike Price : ${K}")
st.write(f"Risk Free Interest : {r}")
st.write(f"Volatility of Underlying : {sigma_stationary}")
#create a 10x10 grid with chosen values from st.form
sigma_range = np.linspace(sigma[0], sigma[1], 10)
spot_range = np.linspace(spot_price_min, spot_price_max, 10)

def plot_option_price_heatmap(calculate_prices_and_greeks, sigma_range, spot_range):
    call_prices = np.zeros((len(sigma_range), len(spot_range)))
    put_prices = np.zeros((len(sigma_range), len(spot_range)))

    for i, sigma in enumerate(np.flip(sigma_range)):
        for j, spot in enumerate(spot_range):
            prices = calculate_prices_and_greeks(spot, K, T, r, sigma)
            call_prices[i, j] = prices["call_price"]
            put_prices[i, j] = prices["put_price"]
            
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.heatmap(call_prices, ax=ax[0], cmap='RdYlGn', annot=True, fmt=".2f",
                xticklabels=np.round(spot_range, 2), yticklabels=np.round(np.flip(sigma_range), 2), )
    ax[0].set_title('Call Option Prices', fontsize = 20)
    ax[0].set_xlabel('Spot Price ($S_0$)', fontsize = 16)
    ax[0].set_ylabel('Volatility ($σ$)', fontsize = 16)

    sns.heatmap(put_prices, ax=ax[1], cmap='RdYlGn', annot=True, fmt=".2f",
                xticklabels=np.round(spot_range, 2), yticklabels=np.round(np.flip(sigma_range), 2))
    ax[1].set_title('Put Option Prices', fontsize = 20)
    ax[1].set_xlabel('Spot Price ($S_0$)', fontsize = 16)
    ax[1].set_ylabel('Volatility ($σ$)', fontsize = 16)

    plt.tight_layout()
    st.pyplot(fig)
plot_option_price_heatmap(calculate_prices_and_greeks, sigma_range, spot_range)

def plot_pnl_heatmap(calculate_prices_and_greeks, sigma_range, spot_range, K, T, r):
    call_pnls = np.zeros((len(sigma_range), len(spot_range)))
    put_pnls = np.zeros((len(sigma_range), len(spot_range)))

    for i, sigma in enumerate(np.flip(sigma_range)):
        for j, spot in enumerate(spot_range):
            prices = calculate_prices_and_greeks(spot, K, T, r, sigma)
            call_price = prices["call_price"]
            put_price = prices["put_price"]
            call_pnls[i, j] = max(0, spot - K) - call_price
            print(call_price)
            put_pnls[i, j] = max(0, K - spot) - put_price

    # Normalize around 0 to split red/green
    norm_call = TwoSlopeNorm(vmin= -20, vcenter=0, vmax=20)
    norm_put = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(call_pnls, ax=ax[0], cmap='RdYlGn', annot=True, fmt=".2f",
                xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(np.flip(sigma_range), 3),
                norm=norm_call)
    ax[0].set_title('Call Option PnL', fontsize = 20)
    ax[0].set_xlabel('Spot Price ($S_0$)', fontsize = 16)
    ax[0].set_ylabel('Volatility ($σ$)', fontsize = 16)
    
    sns.heatmap(put_pnls, ax=ax[1], cmap='RdYlGn', annot=True, fmt=".2f",
                xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(np.flip(sigma_range), 3),
                norm=norm_put)
    ax[1].set_title('Put Option PnL', fontsize = 20)
    ax[1].set_xlabel('Spot Price ($S_0$)', fontsize = 16)
    ax[1].set_ylabel('Volatility ($σ$)', fontsize = 16)

    plt.tight_layout()
    st.pyplot(fig)
plot_pnl_heatmap(calculate_prices_and_greeks, sigma_range, spot_range, K, T, r)
