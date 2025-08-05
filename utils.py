"""def fetch_data(self):
        request_params = StockBarsRequest(
            symbol_or_symbols=[self.ticker],
            timeframe=TimeFrame.Day, # type:ignore
            #YYYY-MM-DD format
            start=datetime(date_range[0].year, date_range[0].month, date_range[0].day),# type:ignore
            end=datetime(date_range[1].year, date_range[1].month, date_range[1].day),# type:ignore
        ) 
        bars = client.get_stock_bars(request_params)
        df = bars.df #type:ignore
        if df.empty:
            st.warning("No data returned for this ticker.")
            return pd.DataFrame()

        df = df.reset_index()
        df['timestamp'] = df['timestamp'].dt.tz_convert('GMT')
        df['Date'] = df['timestamp'].dt.strftime('%d-%m-%Y')
        df = df[['Date', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        return df
        """
        
"""
with st.form("Input DCF Model Parameters"):
    company = st.text_input("Name of Company", )
    col1, col2 = st.columns(2)
    with col1:
        revenue_growth = st.slider("Revenue Growth Rate (%)", 0.0, 20.0, 5.0, step=0.1, key="revenue_growth")
        ebita_margin = st.slider("EBITA Margin (%)", 0.0, 50.0, 20.0, step=0.1, key="ebitda_margin")
        tax_rate = st.slider("Tax Rate (%)", 0.0, 50.0, 21.0, step=0.1, key="tax_rate")
        capex = st.slider("Capital Expenditures (% of Revenue)", 0.0, 20.0, 10.0, step=0.1, key="capex")
        depreciation = st.slider("Depreciation (% of Revenue)", 0.0, 20.0, 5.0, step=0.1, key="depreciation")
        
        submitted = st.form_submit_button("Finished Editing Parameters")
        
    with col2:
        discount_rate = st.slider("Discount Rate (WACC) (%)", 0.0, 20.0, 8.0, step=0.1, key="discount_rate")
        terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 10.0, 2.0, step=0.1, key="terminal_growth")
        forecast_period = st.slider("Forecast Period (years)", 1, 20, 5, step=1, key="forecast_period")
        working_capital = st.slider("Change in Net Working Capital (% of Revenue)", 0.0, 20.0, 5.0, step=0.1, key="working_capital")
        shares_outstanding = st.slider("Shares Outstanding (millions)", 1, 10000, 2000, step=1, key="shares_outstanding")
        ebit = st.slider("EBIT (% of Revenue)", 0.0, 50.0, 15.0, step=0.1, key="ebit")
        """