import pandas as pd
import numpy as np
from scipy.stats import truncnorm

class DividendInvestmentModel:
    def __init__(self, first_investment, periodic_investment, forecast_period,
                 dividend_yield_mean, dividend_yield_std, dividend_frequency,
                 stock_growth_mean, stock_growth_std, sell_frequency,
                 sell_percentage, reinvest_percentage):
        self.first_investment = first_investment
        self.periodic_investment = periodic_investment
        self.forecast_period = forecast_period  # in months
        self.dividend_yield_mean = dividend_yield_mean
        self.dividend_yield_std = dividend_yield_std
        self.dividend_frequency = dividend_frequency.lower()  # 'monthly', 'quarterly', 'annually'
        self.sell_frequency = sell_frequency.lower()  # 'monthly', 'quarterly', 'annually'
        self.stock_growth_mean = stock_growth_mean
        self.stock_growth_std = stock_growth_std
        self.sell_percentage = sell_percentage / 100  # Convert to decimal
        self.reinvest_percentage = reinvest_percentage / 100  # Convert to decimal
        self.dividend_payment_intervals = {'monthly': 1, 'quarterly': 3, 'annually': 12}
        self.append_investment = False

    def _get_bounded_dividend_yield(self):
        """
        Generate a dividend yield randomly from a truncated normal distribution.
        Bounds: 0% to reasonable upper bound (e.g., 20%).
        """
        lower_bound, upper_bound = 0, 0.20  # 0% to 20%
        a, b = (lower_bound - self.dividend_yield_mean) / self.dividend_yield_std, \
               (upper_bound - self.dividend_yield_mean) / self.dividend_yield_std
        return truncnorm.rvs(a, b, loc=self.dividend_yield_mean, scale=self.dividend_yield_std)

    def _get_stock_growth_rate(self):
        """
        Generate a stock price growth rate from a normal distribution.
        """
        return np.random.normal(self.stock_growth_mean, self.stock_growth_std)

    def generate_forecast(self):
        """
        Generate a DataFrame of investment details including dividends, stock appreciation, and sell logic.
        """
        # Initialize columns
        periods = []
        investment_amounts = self.periodic_investment if isinstance(self.periodic_investment, list) else []
        if investment_amounts == []:
            self.append_investment = True
        total_investments = []
        stock_values = []
        dividend_yields = []
        dividends = []
        sell_proceeds = []
        capital_gains = []
        reinvested_amounts = []
        cumulative_reset_flags = []

        # Initialize running totals
        current_date = self.first_investment
        cumulative_investment = 0
        current_stock_value = 0

        sell_interval = self.dividend_payment_intervals[self.sell_frequency]
        dividend_interval = self.dividend_payment_intervals[self.dividend_frequency]

        # Loop through each month in the forecast period
        for period in range(1, self.forecast_period + 1):
            # Add to period
            periods.append(current_date)
            
            # Add periodic investment
            if self.append_investment:
                investment_amounts.append(self.periodic_investment) 
                cumulative_investment += self.periodic_investment
                current_stock_value += self.periodic_investment
            
            else:
                cumulative_investment +- investment_amounts[period-1]  
                current_stock_value += investment_amounts[period-1]
            
            total_investments.append(cumulative_investment)
            

            # Apply stock price growth
            stock_growth_rate = self._get_stock_growth_rate()
            current_stock_value *= (1 + stock_growth_rate)
            stock_values.append(current_stock_value)

            # Handle dividend payments based on frequency
            if period % dividend_interval == 0:
                dividend_yield = self._get_bounded_dividend_yield()
                dividend = current_stock_value * dividend_yield
            else:
                dividend_yield = np.nan
                dividend = np.nan
            dividend_yields.append(dividend_yield)
            dividends.append(dividend)

            # Handle selling logic based on frequency
            if period % sell_interval == 0:
                sell_amount = current_stock_value * self.sell_percentage
                invested_portion_sold = cumulative_investment * self.sell_percentage
                capital_gain = sell_amount - invested_portion_sold

                # Reinvestment logic
                reinvest_amount = sell_amount * self.reinvest_percentage
                cumulative_investment = reinvest_amount  # Reset cumulative investment to reinvested amount
                current_stock_value = reinvest_amount  # Reset stock value

                # Record values
                sell_proceeds.append(sell_amount)
                capital_gains.append(capital_gain)
                reinvested_amounts.append(reinvest_amount)
                cumulative_reset_flags.append(1)
            else:
                sell_proceeds.append(np.nan)
                capital_gains.append(np.nan)
                reinvested_amounts.append(np.nan)
                cumulative_reset_flags.append(0)

            # Increment date by one month
            current_date += pd.DateOffset(months=1)

        # Create DataFrame
        forecast_df = pd.DataFrame({
            'Period': periods,
            'Investment Amount': investment_amounts,
            'Total Investment': np.round(total_investments,2),
            'Stock Value': np.round(stock_values,2),
            'Sell Proceeds': np.round(sell_proceeds,2),
            'Capital Gain': np.round(capital_gains,2),
            'Reinvested Amount': np.round(reinvested_amounts,2),
            'Dividend Yield': dividend_yields,
            'Dividend': np.round(dividends,2),
            'Cumulative Reset': cumulative_reset_flags
        })
        
        forecast_df['Period'] = pd.to_datetime(forecast_df['Period']).dt.strftime('%Y-%m-%d')

        return forecast_df
    
    def transform_to_cashflow(self, forecast_df):
        """
        Transform the forecast DataFrame into a cashflow format with Revenue and Expense.
        Revenue = Dividend + Sell Proceeds
        Expense = Investment Amount + Reinvested Amount
        """
        cashflow_df = pd.DataFrame({
            'Period': forecast_df['Period'],
            'Revenue': forecast_df[['Dividend', 'Sell Proceeds']].sum(axis=1, skipna=True),
            'Expense': forecast_df[['Investment Amount', 'Reinvested Amount']].sum(axis=1, skipna=True)
        })
        return cashflow_df



