import os
os.environ["YFINANCE_NO_CACHE"] = "1"
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI
import re

DIVIDER_COLOUR = "blue"
API_KEY = st.secrets["OpenAI"]["api_key"]

st.set_page_config(page_title="DCF Model",
                page_icon="📝",
                initial_sidebar_state="auto",
                layout="wide"
                )

st.title("Discounted Cash Flow (DCF) Model")
st.subheader("A Discounted Cash Flow (DCF) model is a valuation method that measures the intrinsic value of a company based on the sum of the present value of its future cash. The DCF model is particularly useful for valuing companies with predictable cash flows, such as mature businesses or those in stable industries. The formula for the DCF model is as follows:", divider=DIVIDER_COLOUR)
st.latex(r"""
    DCF = \sum_{t=1}^{n} \frac{CF_t}{(1 + r)^t}
""")
st.write("Where:")
st.latex(r"""
    CF_t = \text{Cash Flow in year } t \\
    r = \text{Discount Rate} \\
    n = \text{Number of years}
""")

st.write("---")
st.header("Input Parameters", divider=DIVIDER_COLOUR)
st.write("In this example we will be using data from NextEra Energy (NEE) to demonstrate the DCF model. You can change the parameters below to see how they affect the DCF valuation.")
st.write("For this model we will use five years of historic data (2020-2024) and we be forecasting the next five years (2025-2029).")
st.markdown("""
            The following steps are required to complete the DCf model:
            1. Project future cash flow (FCF)
            2. Calculate weighted average cost of capital (WACC)
            3. Calculate Terminal Value (TV)
            4. Discount Back to present value
            5. Calculate implied share price
            """)

st.header("Historical and Forecasted Data", divider=DIVIDER_COLOUR)
st.write("All data is taken from Macrotrends and is in Millions of dollars. Please feel free to edit the table below to see how it impacts values.")

input_rows = ["Revenue", "% Growth", "EBIT", "EBIT % of Revenue", "Taxes", "Taxes % of EBIT", "D&A", "D&A % of Revenue", "CapEx", "CapEx % of Revenue","Change in NWC", "Change in NWC % of Revenue"]
#income statement - np.nan will be dynamically calculated
input_data = {
    "2020" : [17997, np.nan, 5116,np.nan, 44,np.nan, 4315, np.nan, 6680, np.nan, -604, np.nan],
    "2021" :[17069, np.nan, 2913, np.nan, 348,np.nan, 4214,np.nan, 7570, np.nan, -486, np.nan],
    "2022" : [20956, np.nan, 4081, np.nan, 586,np.nan, 4790,np.nan, 9185, np.nan, -412, np.nan],
    "2023" : [28144, np.nan, 10237, np.nan, 1006,np.nan, 6151,np.nan, 9400, np.nan,-1393, np.nan],
    "2024" : [24753, np.nan, 7479, np.nan, 339,np.nan, 5761,np.nan, 8200, np.nan, 160, np.nan],
    "2025" : [28466, np.nan, 7401,	np.nan,	740, np.nan, 6191,np.nan, 8500, np.nan, np.nan, np.nan],
    "2026" : [31882, np.nan, 8289,	np.nan,	828, np.nan, 6654, np.nan, 8800, np.nan, np.nan, np.nan],
    "2027" : [35070, np.nan, 9118, np.nan, 911, np.nan, 7152,np.nan, 9100, np.nan, np.nan, np.nan],
    "2028" : [37876, np.nan, 9848, np.nan, 985, np.nan, 7688,np.nan, 9450, np.nan, np.nan, np.nan],
    "2029" : [40149, np.nan, 10438, np.nan, 1044, np.nan, 8262,np.nan, 9800, np.nan, np.nan, np.nan],
}

df_inputs = pd.DataFrame(input_data, index = input_rows)

#fill in the np.nan cells
revenues = df_inputs.loc["Revenue"]
df_inputs.loc["% Growth", revenues.index[1:]] = (revenues.pct_change().iloc[1:]*100).round(2)

ebit = df_inputs.loc["EBIT"]
df_inputs.loc["EBIT % of Revenue"] = ((ebit/revenues)*100).round(2)

# % of EBIT = Taxes/EBIT *100
taxes = df_inputs.loc["Taxes"]
df_inputs.loc["Taxes % of EBIT"] = ((taxes/ebit)*100).round(2)

depreciaton = df_inputs.loc["D&A"]
df_inputs.loc["D&A % of Revenue"] = ((depreciaton/revenues)*100).round(2)

capex = df_inputs.loc["CapEx"]
df_inputs.loc["CapEx % of Revenue"] = ((capex/revenues)*100).round(2)

nwc = df_inputs.loc["Change in NWC"]
df_inputs.loc["Change in NWC % of Revenue"] = ((nwc/revenues)*100).round(2)
st.dataframe(df_inputs)

st.header("1. Project Future Cash Flow", divider=DIVIDER_COLOUR)
st.write("First we need to the unlevered future cash flow. To calculate this we use the following formula:")
st.latex(r"""
\text{FCF} = \text{EBITA} + \text{D\&A} - \text{CapEx} - \Delta\: \text{NWC}
""")

#get years 2025-2029 for columns
years = [str(y) for y in range(2025, 2030)]
output_rows = ["EBITA", "Change in NWC", "Change in NWC % of Revenue", "Unlevered FCF"]

outputs = pd.DataFrame(np.nan, index=output_rows, columns=years)
outputs.loc["EBITA"] = (ebit - taxes)
outputs.loc["Change in NWC % of Revenue"] = np.mean(df_inputs.iloc[-1, 2:5])
outputs.loc["Change in NWC"] = (revenues * outputs.loc["Change in NWC % of Revenue"]).round(2)
outputs.loc["Unlevered FCF"] = (outputs.loc["EBITA"] + depreciaton - capex - (outputs.loc["Change in NWC"]/100)).round(2)
st.dataframe(outputs)

st.write("Change in NWC % of Revenue is derived from the average of the last three years of historical data.")
st.header("2. Calculate the WACC", divider=DIVIDER_COLOUR)
st.write("To calculate this we use the following formula:")
st.latex(r"""
    \text{WACC} = \text{\% Equity} \times \text{Cost of Equity} + (\text{\% Debt}\times \text{Cost of Debt}\times (\text{1 - tax rate}))
""")
st.write("Where:")
st.latex(r"""
    \text{Cost of Equity} = \text{Risk Free Rate} + (\text{Beta} \times (\text{Expected Market Return - Risk Free Rate}))
""")

ticker = yf.Ticker("NEE")#type:ignore
current_stock_price = ticker.history(period = "1d")["Close"].iloc[-1]
shares_outstanding = ticker.info.get("sharesOutstanding", None)
market_cap = ticker.info.get("marketCap", None)

col1, col2 = st.columns(2, border=True)
with col1:
    with st.form("WACC Calculator", border=False):
        st.write("WACC Calculator - All values in $ Millions")
        company_debt = st.number_input("Company Debt", value=float(89670), format="%0.2f")
        company_cash = st.number_input("Company Cash", value= float(2419))
        equity_value = st.number_input(f"Equity Value (Market Cap)", value=float(market_cap/1_000_000), format="%0.2f")#type:ignore
        cost_of_debt = st.number_input("Cost of Debt (%)", value = 5.0)
        tax_rate = st.number_input("Effective Tax Rate (%)", value= 5.5)
        risk_free_rate = st.number_input("Risk Free Rate (%)", value=4.5)
        beta = st.number_input("Beta", value =0.5)
        market_risk_premium = st.number_input("Market Risk Premium (%)", value=5.0)
        submitted = st.form_submit_button("Submit Values")
        
with col2:
    debt_plus_equity = equity_value + company_debt
    percent_debt = ((company_debt/debt_plus_equity)*100)
    percent_equity = (equity_value/(debt_plus_equity)*100)
    cost_of_equity = (beta*market_risk_premium)+risk_free_rate
    wacc = (percent_equity/100) * cost_of_equity + ((percent_debt/100) * cost_of_debt * (1 - (tax_rate/100)))
    
    st.header(f"Debt + Equity: ${debt_plus_equity:,.2f}")
    st.subheader("")
    st.header(f"% Debt: {percent_debt:.4}%")
    st.subheader("")
    st.header(f"% Equity: {percent_equity:,.2f}%")
    st.subheader("")
    st.header(f"Cost of Equity: {cost_of_equity:.3}%")
    st.subheader("")
    st.header(f"WACC: {wacc:.3}%")

st.header("3. Calculating Terminal Value", divider=DIVIDER_COLOUR)
st.write("The Terminal Value (TV) is the value of company's future free cash flow from it's last projected year until the end of time. We will use the perpetuity growth method described by using the WACC, FCF and terminal growth rate (TGR). The terminal growth rate is generally based around GDP% of the country the business is located and is a highly sensitive variable. The TV is calculated by the following formula: ")
col1, col2 = st.columns(2)
col1.latex(r"""
        \frac{\text{Last Year FCF}\times\text{(1 + TGR)}}{\text{(WACC - TGR)}}
        """)
with col2:
    tgr = st.slider("Terminal Growth Rate (%)", min_value=0.0, max_value=5.0, value=2.5, step=0.05)

last_year_fcf = outputs.iloc[-1, -1]

terminal_value = (last_year_fcf * (1 + (tgr/100)))/((wacc/100)-(tgr/100))#type:ignore
st.subheader(f"Terminal Value: ${terminal_value/1000:,.2f} Billion")
st.write("---")
st.header("4. Discount Back to Present Value (PV)", divider=DIVIDER_COLOUR)

st.latex(r"""
        \text{PV} = \frac{\text{FCF for year} \:t}{(\text{1 + WACC})^{t}}
        """)

col_length = range(1,len(years)+1)
outputs.loc["Present Value of FCF"] = outputs.loc["Unlevered FCF"]/((1+(wacc/100))** pd.Series(col_length, index = outputs.columns))
st.dataframe(outputs)
present_value_of_termial_value = terminal_value/((1+(wacc/100))**5)
st.subheader(f"Present Value of Terminal Value: ${present_value_of_termial_value/1000:,.3f} Billion")
st.write("---")

st.header("5. Calculate implied share price",divider=DIVIDER_COLOUR)
enterprise_value = outputs.loc["Present Value of FCF"].sum() + present_value_of_termial_value
st.subheader(f"Enterprise Value: ${enterprise_value/1000:,.2f} Billion")
calculated_equity_value = enterprise_value +company_cash - company_debt
st.subheader(f"Calculated Equity Value: ${calculated_equity_value/1000:,.4} Billion")

share_price = calculated_equity_value/float(shares_outstanding/1_000_000)#type:ignore
st.subheader(f"Calculated Share Price: ${share_price:,.4}")
st.subheader(f"Actual Share Price: $71.97")


st.write("---")
st.header("AI powered DCF Insights", divider=DIVIDER_COLOUR)
st.write("This section will provide AI-generated insights based on the DCF model parameters. The insights will be generated using a language model to help you understand the implications of the inputs on the DCF valuation.")

left, middle, right = st.columns(3)

if middle.button("Generate AI Insights"):
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages = [
        {
        "role": "system",
        "content": "You are a thoughtful and clear AI financial analyst. Help the user understand what a Discounted Cash Flow (DCF) model says about a company’s future. Focus on what the numbers mean — not just how the model works, but what it says about growth, risk, and value. Use the data provided (revenue, free cash flow, growth rate, WACC, terminal value) to explain where the business might be headed, and what could change that picture. Keep it simple and focused. Help the user think like a long-term investor — spotting what matters and questioning assumptions. Give answer in plain text."
        }
        ,
        
        {"role": "user", "content": f"Here is the DCF valuation summary for NextEra Energy:\n"
        f"Company Debt : {company_debt}\n"
        f"Company Cash : {company_cash}\n"
        f"Equity Value (Market Cap) : {equity_value}\n"
        f"Cost of Debt (%) : {cost_of_debt}\n"
        f"Effective Tax Rate (%): {tax_rate}\n"
        f"Risk Free Rate (%) : {risk_free_rate}\n"
        f"Beta : {beta}\n"
        f"Market Risk Premium (%) : {market_risk_premium}\n"
        f"Debt + Equity : ${debt_plus_equity}\n"
        f"Percent Debt (%) : {percent_debt}\n"
        f"Percent Equity (%) : {percent_equity}\n"
        f"Cost of Equity (%) : {cost_of_equity}\n"
        f"WACC (%) : {wacc}\n"
        f"Terminal Value : ${terminal_value}\n"
        f"Present Value of Terminal Value : ${present_value_of_termial_value}\n"
        f"Enterprise Value : ${enterprise_value}\n"
        f"Calculated Equity Value : ${calculated_equity_value}\n"
        f"Calculated Share Value : ${share_price}\n"
        f"Todays Share Price : ${current_stock_price:,.3f} \n"
        f"Unlevered FCF (2025-2029) : {outputs.loc['Unlevered FCF']}\n"
        f"Present Value of FCF (2025-2029) : {outputs.loc['Present Value of FCF']}\n"
        }
        ]
    )

#prevents text coming out italic
    def escape_markdown(text):
        # Escape underscores, asterisks, and dollar signs
        text = re.sub(r'(?<!\\)_', r'\_', text)
        text = re.sub(r'(?<!\\)\$', r'\$', text)
        return text

    safe_content = escape_markdown(response.choices[0].message.content)
    st.markdown(safe_content)

