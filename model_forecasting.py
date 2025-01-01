import pandas as pd
import numpy as np
import plotly.express as px
import numpy_financial as npf
import plotly.graph_objects as go
import random
import calendar
from datetime import datetime, date, timedelta




class ModelForecastPerformance:
    def __init__(self, company_variables = None):
        
        company_variables = company_variables or {}
        
        self.ebit = company_variables.get('EBIT', 0)
        self.net_sales_growth = company_variables.get('Net Sales Growth', 0)
        self.relative_variability_net_sales = company_variables.get('Relative Variation of Net Sales', 0)
        self.ebit_ratio = company_variables.get('EBIT Ratio', 0)
        self.general_expense = -company_variables.get('General Expense', 0)



    def forecast_indirect_cost(self, number_of_forecasted_periods, start_year=2024, start_month=1):
        indirect_cost = self.general_expense / 12
        periods = np.arange(start_month, number_of_forecasted_periods + start_month)
        start_year = start_year
        
        periods = [self.generate_date_from_month(int(period), method='last_day', start_year=start_year) for period in periods]
        
        # Create a DataFrame for the result
        forecast_df = pd.DataFrame({
            'Period': periods,
            'Revenue': 0,
            'Expense': np.round(indirect_cost, 0)  # Round Revenue values to 2 decimal places
        })
        
        forecast_df['Period'] = pd.to_datetime(forecast_df['Period'], format='%d-%m-%Y')
        forecast_df['Period'] = forecast_df['Period'].dt.date
        forecast_df = forecast_df.sort_values(by='Period').reset_index(drop=True)
        
        
        return forecast_df
        


    def forecast_revenue_expenses(self, number_of_forecasted_periods, start_month=1):
        """
        Generates a DataFrame of forecasted Revenue (Net Sales) and Expenses (COGS) values.
        
        Parameters:
        number_of_forecasted_periods: Number of forecast periods (months).
        
        Returns:
        DataFrame with three columns: 'Period', 'Revenue', and 'Expenses'.
        """
        
        
        # Initial EBIT, revenue growth, and indirect cost per month
        initial_EBIT = self.ebit / 12  # Initial monthly EBIT
        ebit_growth_monthly = self.net_sales_growth / 12  # Monthly net sales growth
        relative_variation = self.relative_variability_net_sales
        ebit_ratio = self.ebit_ratio
        indirect_cost = self.general_expense / 12  # Monthly indirect costs
        
        growth = 0
        
        minimum_growth = self.net_sales_growth * (1 - (self.net_sales_growth * 0.1)) * (number_of_forecasted_periods/12)

        while growth < minimum_growth:
            # Create an array for the period
            periods = np.arange(start_month, number_of_forecasted_periods + start_month)
            
            # Calculate the deterministic growth component (trend)
            trend = initial_EBIT * (1 + ebit_growth_monthly) ** periods
            
            # Introduce random variation based on the relative variation (CV)
            # Coefficient of variation is std/mean, so we introduce noise with appropriate std
            random_variation = np.random.normal(loc=0, scale=relative_variation, size=number_of_forecasted_periods)
            
            # Combine the trend with the variation, ensuring positive EBIT values
            ebit_values = trend * (1 + random_variation)
            

            total_ebit_after = np.sum(ebit_values) * 12 / number_of_forecasted_periods
            # print("Total EBIT after: ", total_ebit_after)
            growth = (total_ebit_after - self.ebit) / self.ebit
            # print("New EBIT", total_ebit_after)
            # print("Old EBIT", self.ebit)
            # print("Minimum Growth", minimum_growth)
            # print("Growth: ", growth)
            
            mean_ebit_values = np.mean(ebit_values)
            std_ebit_values = np.std(ebit_values)
            
            # print("Coefficient of Variation: ", std_ebit_values / mean_ebit_values)
            
            # print(self.net_sales_growth)

        # # Apply net sales growth to EBIT for cumulative effect
        # total_growth_factor = (1 + self.net_sales_growth) ** number_of_forecasted_periods
        # adjusted_ebit_values = ebit_values * total_growth_factor
        
        # Calculate Revenue (Net Sales) and Expenses (COGS)
        revenue_values = ebit_values / ebit_ratio  # Net sales derived from EBIT
        expense_values = revenue_values - ebit_values - indirect_cost  # COGS derived from revenue minus EBIT and indirect costs
    
        
        
        
        # Create a DataFrame for the result
        forecast_df = pd.DataFrame({
            'Period': periods,
            'Revenue': np.round(revenue_values, 0),  # Rounded Revenue values (Net Sales)
            'Expense': np.round(expense_values, 0)   # Rounded Expense values (COGS)
        })
        
        return forecast_df


    
    def forecast_revenue_expenses_MA(self, historical_data_df, number_of_forecasted_periods, MA_period, start_month=1):
        """
        Generates a forecast of Revenue and Expenses using a Simple Moving Average (SMA) approach.
        
        Parameters:
        historical_data_df: DataFrame containing historical data with columns 'Period', 'Revenue', 'Expenses'.
        number_of_forecasted_periods: Number of forecast periods (months).
        MA_period: Number of periods to use for calculating the moving average.
        
        Returns:
        DataFrame with forecasted 'Period', 'Revenue', and 'Expenses'.
        """
        
        # Extract the historical Revenue and Expenses
        historical_revenue = historical_data_df['Revenue'].values
        historical_expenses = historical_data_df['Expense'].values
        
        # Initialize forecast lists with the historical data for reference
        revenue_forecast = list(historical_revenue)
        expenses_forecast = list(historical_expenses)
        
        # Generate forecast for the desired number of periods
        for period in range(number_of_forecasted_periods):
            # For the first forecasted period, use the last MA_period rows of historical data
            if period == 0:
                revenue_ma = np.mean(revenue_forecast[-MA_period:])
                expenses_ma = np.mean(expenses_forecast[-MA_period:])
            else:
                # For subsequent periods, include the forecasted values in the moving average
                revenue_ma = np.mean(revenue_forecast[-MA_period:])
                expenses_ma = np.mean(expenses_forecast[-MA_period:])
            
            # Append the calculated forecast values to the forecast list
            revenue_forecast.append(revenue_ma)
            expenses_forecast.append(expenses_ma)
        
        # Create a forecast DataFrame, excluding the initial historical data
        forecast_periods = np.arange(start_month, number_of_forecasted_periods + start_month)
        forecast_df = pd.DataFrame({
            'Period': forecast_periods,
            'Revenue': np.round(revenue_forecast[-number_of_forecasted_periods:], 0),
            'Expense': np.round(expenses_forecast[-number_of_forecasted_periods:], 0)
        })
        
        return forecast_df

    
    def plot_ebit_line_chart(self, df):
        # Create a line plot using Plotly
        fig = px.line(df, x='Period', y='EBIT', markers=True)
        
        # Update layout for better presentation
        fig.update_layout(
            xaxis_title='Period',
            yaxis_title='EBIT',
            template='plotly_white'
        )
        
        fig.update_layout(yaxis=dict(range=[0, 100000]))
        
        # Return the figure object
        return fig
    
    def merge_and_plot_ebit(self, df1, df2):

        
        def process_dataframe(df):
            # Check if the dataframe contains Revenue and Expense, then calculate Profit
            if 'Revenue' in df.columns and 'Expense' in df.columns:
                df['EBIT'] = df['Revenue'] - df['Expense']
                df = df.drop(columns=['Revenue', 'Expense'])
            return df
        
        # Process both dataframes
        df1 = process_dataframe(df1)
        df2 = process_dataframe(df2)
        
        # Combine the dataframes by Period and sum the Profit
        combined_df = pd.concat([df1, df2]).groupby('Period', as_index=False).sum()
        
        # Create the line chart using Plotly
        fig = px.line(combined_df, x='Period', y='EBIT', markers=True)
        
        # Update layout for better presentation
        fig.update_layout(
            xaxis_title='Period',
            yaxis_title='EBIT',
            template='plotly_white'
        )
        
        # Set y-axis range (this can be adjusted as per the data)
        fig.update_layout(yaxis=dict(range=[0, combined_df['EBIT'].max() * 1.1]))
        
        return combined_df, fig
    
    
    def compare_and_plot_ebit(self, df1, df2, label=['DataFrame 1', 'DataFrame 2']):
        """
        Process two DataFrames to ensure they both have an 'EBIT' column, 
        then generate a Plotly line chart comparing them.

        Parameters:
        df1: First input DataFrame (Period, Revenue, Expense) or (Period, EBIT).
        df2: Second input DataFrame (Period, Revenue, Expense) or (Period, EBIT).

        Returns:
        fig: Plotly figure showing comparison of EBIT over Period for both DataFrames.
        """
        
        def process_dataframe(df):
            # Check if the dataframe contains Revenue and Expense, then calculate EBIT
            if 'Revenue' in df.columns and 'Expense' in df.columns:
                df['EBIT'] = df['Revenue'] - df['Expense']
                df = df.drop(columns=['Revenue', 'Expense'])
            return df
        
        # Process both dataframes
        df1 = process_dataframe(df1)
        df2 = process_dataframe(df2)
        
        # Add a label to distinguish the two dataframes in the plot
        df1['Label'] = label[0]
        df2['Label'] = label[1]
        
        # Combine the dataframes without merging, just concatenating them for comparison
        combined_df = pd.concat([df1, df2])
        
        # Create the line chart using Plotly with different colors for each DataFrame
        fig = px.line(combined_df, x='Period', y='EBIT', color='Label', markers=True)
        
        # Update layout for better presentation
        fig.update_layout(
            xaxis_title='Period',
            yaxis_title='EBIT',
            template='plotly_white'
        )
        
        return fig

    # Placeholder example usage of the function. Uncomment and modify to test with real dataframes.
    # df1 = pd.DataFrame({'Period': [1, 2, 3], 'Revenue': [200, 300, 400], 'Expense': [50, 60, 70]})
    # df2 = pd.DataFrame({'Period': [1, 2, 3], 'EBIT': [120, 150, 180]})
    # fig = compare_and_plot_data(df1, df2)
    # fig.show()
    
    def loan_amortization_schedule(self, principal, annual_interest_rate, loan_term_years, start_year=2024):
        """
        Generates an amortization schedule for a loan and calculates key figures.
        
        Parameters:
        principal: The loan amount (principal).
        annual_interest_rate: The annual interest rate as a percentage (e.g., 5 for 5%).
        loan_term_years: The loan term in years.
        
        Returns:
        amortization_df: DataFrame with amortization schedule columns (Month, Monthly Payment, Principal Payment, Interest Payment, Remaining Principal).
        monthly_payment: The fixed monthly payment.
        total_principal: The total principal paid (same as initial principal).
        total_interest: The total interest paid over the loan term.
        """
        
        # Convert annual interest rate to monthly interest rate
        monthly_interest_rate = annual_interest_rate / 100 / 12
        
        # Calculate the total number of payments (loan term in months)
        total_payments = loan_term_years * 12
        
        # Calculate the fixed monthly payment using the annuity formula
        monthly_payment = npf.pmt(monthly_interest_rate, total_payments, -principal)
        
        # Initialize lists to store amortization data
        month_list = []
        payment_list = []
        principal_payment_list = []
        interest_payment_list = []
        remaining_principal_list = []
        
        remaining_principal = principal
        total_interest_paid = 0
        
        # Generate the amortization schedule
        for month in range(1, total_payments + 1):
            # Interest payment for this month
            interest_payment = remaining_principal * monthly_interest_rate
            # Principal payment for this month
            principal_payment = monthly_payment - interest_payment
            # Update remaining principal
            remaining_principal -= principal_payment
            
            # Append data to lists
            month_list.append(month)
            payment_list.append(round(monthly_payment, 2))
            principal_payment_list.append(round(principal_payment, 2))
            interest_payment_list.append(round(interest_payment, 2))
            remaining_principal_list.append(round(remaining_principal, 2))
            
            # Accumulate total interest
            total_interest_paid += interest_payment
        
        
        month_list = [self.generate_date_from_month(int(period), method='first_day', start_year=start_year) for period in month_list]
        
        # Create DataFrame for amortization schedule
        amortization_df = pd.DataFrame({
            'Period': month_list,
            'Monthly Payment': payment_list,
            'Principal Payment': principal_payment_list,
            'Interest Payment': interest_payment_list,
            'Remaining Principal': remaining_principal_list
        })
        
        # Calculate total interest paid over the loan term
        total_interest = round(total_interest_paid, 2)
        
        amortization_df['Period'] = pd.to_datetime(amortization_df['Period'], format='%d-%m-%Y')
        amortization_df['Period'] = amortization_df['Period'].dt.date
        amortization_df = amortization_df.sort_values(by='Period').reset_index(drop=True)
        
        return amortization_df, round(monthly_payment, 2), round(principal, 2), total_interest
    
    def generate_forecast_treatment_df(self, treatment_df, cashflow_df, basis="Australia", tolerance=0.05):
        """
        Generates a DataFrame of randomly selected treatments for each period in cashflow_df,
        until the accumulated revenue and expense match the cashflow values within the specified tolerance.
        
        Parameters:
        treatment_df: DataFrame containing treatments with revenue and expense in AUD and IDR.
        cashflow_df: DataFrame containing the cashflow for each period with columns Period, Revenue, Expense.
        basis: "Australia" or "Indonesia", determines whether to use AUD or IDR for revenue and expense.
        tolerance: The tolerance level for revenue and expense matching (as a fraction, e.g., 0.1 for 10%).
        
        Returns:
        forecast_treatment_df: DataFrame with columns Period, Treatment, Revenue, Expense.
        """
        
        # Define which columns to use for revenue and expense based on the basis
        if basis == "Australia":
            revenue_col = 'total_price_AUD'
            expense_col = 'total_cost_AUD'
        elif basis == "Indonesia":
            revenue_col = 'total_price_IDR'
            expense_col = 'total_cost_IDR'
        else:
            raise ValueError("Basis must be 'Australia' or 'Indonesia'")
        
        # Initialize list to store forecast data
        forecast_data = []

        # Loop through each period in cashflow_df
        for i, row in cashflow_df.iterrows():
            period = row['Period']
            target_revenue = row['Revenue']
            target_expense = row['Expense']
            
            # Calculate the tolerance ranges
            revenue_lower_bound = target_revenue * (1 - tolerance)
            revenue_upper_bound = target_revenue * (1 + tolerance)
            expense_lower_bound = target_expense * (1 - tolerance)
            expense_upper_bound = target_expense * (1 + tolerance)
            
            # Initialize accumulators for revenue and expense
            accumulated_revenue = 0
            accumulated_expense = 0
            
            # Keep track of treatments that caused breaches to avoid resampling them
            discarded_treatments = set()
            
            # Select random treatments until revenue and expense are within the tolerance
            while True:
                # Randomly select a treatment, making sure it hasn't been discarded
                available_treatments = treatment_df[~treatment_df['Treatment'].isin(discarded_treatments)]
                
                # If no valid treatments remain, break the loop
                if available_treatments.empty:
                    break
                
                selected_treatment = available_treatments.sample(n=1).iloc[0]
                
                treatment_revenue = selected_treatment[revenue_col]
                treatment_expense = selected_treatment[expense_col]
                
                # Check if adding this treatment would exceed the revenue or expense limits
                new_accumulated_revenue = accumulated_revenue + treatment_revenue
                new_accumulated_expense = accumulated_expense + treatment_expense
                
                # Check if it breaches revenue limit
                if new_accumulated_revenue > revenue_upper_bound:
                    discarded_treatments.add(selected_treatment['Treatment'])
                    continue  # Resample, this treatment breaches revenue upper bound
                
                # Check if it breaches expense limit
                if new_accumulated_expense > expense_upper_bound:
                    discarded_treatments.add(selected_treatment['Treatment'])
                    continue  # Resample, this treatment breaches expense upper bound
                
                # If it passes both checks, add the treatment to the forecast
                accumulated_revenue = new_accumulated_revenue
                accumulated_expense = new_accumulated_expense
                
                forecast_data.append({
                    'Period': period,
                    'Treatment': selected_treatment['Treatment'],
                    'Revenue': treatment_revenue,
                    'Expense': treatment_expense
                })
                
                # Check if the accumulated revenue and expense are within their bounds
                if (revenue_lower_bound <= accumulated_revenue <= revenue_upper_bound and
                    expense_lower_bound <= accumulated_expense <= expense_upper_bound):
                    break  # Terminate once both are within the tolerance range
        
        # Create a DataFrame from the forecast data
        forecast_treatment_df = pd.DataFrame(forecast_data)
        
        return forecast_treatment_df
    
    def total_days_from_start(self, number_of_months, start_year=2024, start_month=1):
        # Predefined start year and month
        start_year = start_year
        start_month = start_month

        total_days = 0
        current_year = start_year
        current_month = start_month

        for _ in range(number_of_months):
            # Get the number of days in the current month and year
            days_in_month = calendar.monthrange(current_year, current_month)[1]
            total_days += days_in_month

            # Move to the next month
            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1

        return total_days


    def generate_forecast_treatment_df_by_profit(self, treatment_df, cashflow_df, basis="Australia", tolerance=0.1, number_of_unique_patient=1000, start_year=2024, patient_pool=None):
        """
        Generates a DataFrame of randomly selected treatments for each period in cashflow_df,
        until the accumulated profit (Revenue - Expense) matches the cashflow profit within the specified tolerance.
        
        Parameters:
        treatment_df: DataFrame containing treatments with revenue and expense in AUD and IDR.
        cashflow_df: DataFrame containing the cashflow for each period with columns Period, Revenue, Expense.
        basis: "Australia" or "Indonesia", determines whether to use AUD or IDR for revenue and expense.
        tolerance: The tolerance level for profit matching (as a fraction, e.g., 0.1 for 10%).
        number_of_unique_patient: The number of unique customers to generate, default is 1000.
        
        Returns:
        forecast_treatment_df: DataFrame with columns Period, Treatment, Revenue, Expense, Customer ID.
        """
        
        # Define which columns to use for revenue and expense based on the basis
        if basis == "Australia":
            revenue_col = 'total_price_AUD'
            expense_col = 'total_cost_AUD'
        elif basis == "Indonesia":
            revenue_col = 'total_price_IDR'
            expense_col = 'total_cost_IDR'
        else:
            raise ValueError("Basis must be 'Australia' or 'Indonesia'")
        
        # Add a Profit column to cashflow_df
        cashflow_df['Profit'] = cashflow_df['Revenue'] - cashflow_df['Expense']
        
        # Generate a list of unique Customer IDs
        customer_ids = [f'Customer {i}' for i in range(1, number_of_unique_patient + 1)]
        
        # Initialize list to store forecast data
        forecast_data = []

        # Loop through each period in cashflow_df
        for i, row in cashflow_df.iterrows():
            period = row['Period']
            target_profit = row['Profit']
            
            # Calculate the tolerance range for profit
            profit_lower_bound = target_profit * (1 - tolerance)
            profit_upper_bound = target_profit * (1 + tolerance)
            
            # Initialize accumulators for revenue and expense
            accumulated_revenue = 0
            accumulated_expense = 0
            
            # Keep track of treatments that caused breaches to avoid resampling them
            discarded_treatments = set()
            
            # Select random treatments until profit is within the tolerance
            while True:
                # Randomly select a treatment, making sure it hasn't been discarded
                available_treatments = treatment_df[~treatment_df['Treatment'].isin(discarded_treatments)]
                
                # If no valid treatments remain, break the loop
                if available_treatments.empty:
                    break
                
                selected_treatment = available_treatments.sample(n=1).iloc[0]
                
                treatment_revenue = selected_treatment[revenue_col]
                treatment_expense = selected_treatment[expense_col]
                
                # Check if adding this treatment would exceed the profit upper limit
                new_accumulated_revenue = accumulated_revenue + treatment_revenue
                new_accumulated_expense = accumulated_expense + treatment_expense
                new_accumulated_profit = new_accumulated_revenue - new_accumulated_expense
                
                # If the accumulated profit exceeds the upper limit, discard the treatment and resample
                if new_accumulated_profit > profit_upper_bound:
                    discarded_treatments.add(selected_treatment['Treatment'])
                    continue  # Resample, this treatment breaches profit upper bound
                
                # If it passes the check, add the treatment to the forecast
                accumulated_revenue = new_accumulated_revenue
                accumulated_expense = new_accumulated_expense
                
                # Randomly select a Customer ID for this row
                customer_id = np.random.choice(customer_ids)
                
                forecast_data.append({
                    'Period': self.generate_date_from_month(int(period), start_year=start_year),
                    'Treatment': selected_treatment['Treatment'],
                    'Revenue': treatment_revenue,
                    'Expense': treatment_expense,
                    'Customer ID': customer_id
                })
                
                # Check if the accumulated profit is within the tolerance bounds
                accumulated_profit = accumulated_revenue - accumulated_expense
                if profit_lower_bound <= accumulated_profit <= profit_upper_bound:
                    break  # Terminate once the profit is within the tolerance range
        
        # Create a DataFrame from the forecast data
        forecast_treatment_df = pd.DataFrame(forecast_data)
        
        # sort by 'Period' column of date 
        forecast_treatment_df['Period'] = pd.to_datetime(forecast_treatment_df['Period'], format='%d-%m-%Y')
        forecast_treatment_df['Period'] = forecast_treatment_df['Period'].dt.date
        forecast_treatment_df = forecast_treatment_df.sort_values(by='Period').reset_index(drop=True)
        
        
        return forecast_treatment_df

    def generate_forecast_item_code_by_profit(self, item_code_df, cashflow_df, demand_history, basis="Australia", tolerance=0.1, number_of_unique_patient=1000, start_year=2024, patient_pool=None, OHT_salary=35, specialist_salary=125):

        # Note no dentist, only OHT and Specialist
        
        item_code_df['COGS Salary AUD'] = item_code_df.apply(lambda row: OHT_salary * row['duration'] / 60 if row['medical_officer_new'] == 'OHT' else specialist_salary * row['duration'] / 60, axis=1)
        
# Extract and safely format the codes
        demand_codes = demand_history['Code'].values
        demand_codes = [f"{int(code):03}" for code in demand_codes if str(code).isdigit()]

        
        # print(demand_codes)
        # print(f"Number of demand codes: {len(demand_codes)}")
        
        demand_probs = (demand_history['Total Demand'] / demand_history['Total Demand'].sum()).values
        
        # print(demand_probs)
        # print(f"Number of demand probs: {len(demand_probs)}")
        
        
        # Define which columns to use for revenue and expense based on the basis
        if basis == "Australia":
            revenue_col = 'price AUD'
            expense_col = 'cost_material AUD'
            salary_expense_col = 'COGS Salary AUD'
        elif basis == "Indonesia":
            revenue_col = 'price IDR'
            expense_col = 'cost_material IDR'
            salary_expense_col = 'COGS Salary AUD'
        else:
            raise ValueError("Basis must be 'Australia' or 'Indonesia'")
        
        
        
        # Add a Profit column to cashflow_df
        cashflow_df['Profit'] = cashflow_df['Revenue'] - cashflow_df['Expense']
        
        # Generate a list of unique Customer IDs
        customer_ids = [f'Customer {i}' for i in range(1, number_of_unique_patient + 1)] if patient_pool is None else patient_pool
        
        # Initialize list to store forecast data
        forecast_data = []

        # Loop through each period in cashflow_df
        for i, row in cashflow_df.iterrows():
            period = row['Period']
            target_profit = row['Profit']
            
            # Calculate the tolerance range for profit
            profit_lower_bound = target_profit * (1 - tolerance)
            profit_upper_bound = target_profit * (1 + tolerance)
            
            # Initialize accumulators for revenue and expense
            accumulated_revenue = 0
            accumulated_expense = 0
            
            # Keep track of treatments that caused breaches to avoid resampling them
            discarded_treatments = set()
            
            # Select random treatments until profit is within the tolerance
            while True:
                # Randomly select a treatment, making sure it hasn't been discarded
                
                
                # print(f"Unique item codes: {item_code_df['item_number'].unique()}")
                
                filtered_treatments = item_code_df[
                    (item_code_df['item_number'].isin(demand_codes)) &
                    (~item_code_df['item_number'].isin(discarded_treatments))
                ]

                # Step 2: Ensure we have treatments left after filtering
                if filtered_treatments.empty:
                    break

                # Step 3: Map the remaining treatments to their probabilities
                filtered_codes = filtered_treatments['item_number'].values
                filtered_probs = [demand_probs[demand_codes.index(code)] for code in filtered_codes]

                # Normalize probabilities in case some demand_probs are excluded
                filtered_probs = np.array(filtered_probs) / np.sum(filtered_probs)

                # Step 4: Sample one item using the probabilities
                selected_code = np.random.choice(filtered_codes, p=filtered_probs)

                # Step 5: Retrieve the selected treatment row
                selected_treatment = filtered_treatments[filtered_treatments['item_number'] == selected_code].iloc[0]
                
                treatment_revenue = selected_treatment[revenue_col]
                treatment_expense = selected_treatment[expense_col] + selected_treatment[salary_expense_col]
                
                # Check if adding this treatment would exceed the profit upper limit
                new_accumulated_revenue = accumulated_revenue + treatment_revenue
                new_accumulated_expense = accumulated_expense + treatment_expense
                new_accumulated_profit = new_accumulated_revenue - new_accumulated_expense
                
                # If the accumulated profit exceeds the upper limit, discard the treatment and resample
                if new_accumulated_profit > profit_upper_bound:
                    discarded_treatments.add(selected_treatment['item_number'])
                    continue  # Resample, this treatment breaches profit upper bound
                
                # If it passes the check, add the treatment to the forecast
                accumulated_revenue = new_accumulated_revenue
                accumulated_expense = new_accumulated_expense
                
                # Randomly select a Customer ID for this row
                customer_id = np.random.choice(customer_ids)
                
                forecast_data.append({
                    'Period': self.generate_date_from_month(int(period), start_year=start_year),
                    'Treatment': selected_treatment['item_number'],
                    'Revenue': treatment_revenue,
                    'Expense': treatment_expense,
                    'Customer ID': customer_id
                })
                
                # Check if the accumulated profit is within the tolerance bounds
                accumulated_profit = accumulated_revenue - accumulated_expense
                if profit_lower_bound <= accumulated_profit <= profit_upper_bound:
                    break  # Terminate once the profit is within the tolerance range
        
        # Create a DataFrame from the forecast data
        forecast_treatment_df = pd.DataFrame(forecast_data)
        
        # print(forecast_treatment_df)
        
        
        # sort by 'Period' column of date 
        forecast_treatment_df['Period'] = pd.to_datetime(forecast_treatment_df['Period'], format='%d-%m-%Y')
        forecast_treatment_df['Period'] = forecast_treatment_df['Period'].dt.date
        forecast_treatment_df = forecast_treatment_df.sort_values(by='Period').reset_index(drop=True)
        
        
        return forecast_treatment_df




    def generate_date_from_month(self, month, method='random', start_year=2024):
        # Starting year
        start_year = start_year

        # Calculate the year and month from the given month index
        year = start_year + (month - 1) // 12
        month = (month - 1) % 12 + 1  # Ensure month wraps around to a valid 1-12 range

        # Get the number of days in the given month and year
        days_in_month = calendar.monthrange(year, month)[1]

        if method == 'random':
            # Generate a random day within the valid range for the month
            day = random.randint(1, days_in_month)
        elif method == 'last_day':
            # Set the day to the last day of the month
            day = days_in_month
        elif method == 'first_day':
            # Set the day to the first day of the month
            day = 1
        else:
            raise ValueError("Invalid method. Choose 'random', 'last_day', or 'first_day'.")

        # Format the date as dd-mm-yyyy
        generated_date = date(year, month, day).strftime("%d-%m-%Y")
        generated_date = datetime.strptime(generated_date, "%d-%m-%Y")

        return generated_date


    def generate_date_from_month_year(self, month_year, method='random'):
        """
        Generate a date based on the given Month-Year in 'YYYY-MM' format.

        Args:
            month_year (str): The Month-Year string in 'YYYY-MM' format.
            method (str): The method for selecting the day. Options: 'random', 'first_day', 'last_day'.

        Returns:
            datetime: The generated date object.
        """
        # Parse the 'YYYY-MM' format
        year, month = map(int, month_year.split('-'))

        # Get the number of days in the given month and year
        days_in_month = calendar.monthrange(year, month)[1]

        if method == 'random':
            # Generate a random day within the valid range for the month
            day = random.randint(1, days_in_month)
        elif method == 'last_day':
            # Set the day to the last day of the month
            day = days_in_month
        elif method == 'first_day':
            # Set the day to the first day of the month
            day = 1
        else:
            raise ValueError("Invalid method. Choose 'random', 'last_day', or 'first_day'.")

        # Format the date as dd-mm-yyyy
        generated_date = date(year, month, day)

        return generated_date
        
    

    def forecast_revenue_expenses_MA_by_profit(self, historical_data_df, number_of_forecasted_periods, MA_period, treatment_df, basis="Australia", tolerance=0.1, start_month=1):
        """
        Generates a forecast of Revenue and Expenses using a Simple Moving Average (SMA) approach,
        and adjusts based on randomly selected treatments within the specified tolerance.

        Parameters:
        historical_data_df: DataFrame containing historical data with columns 'Period', 'Revenue', and 'Expenses'.
        number_of_forecasted_periods: Number of forecast periods (months).
        MA_period: Number of periods to use for calculating the moving average.
        treatment_df: DataFrame of treatments with columns 'Treatment', 'total_price_AUD', 'total_price_IDR', 'total_cost_AUD', and 'total_cost_IDR'.
        basis: "Australia" or "Indonesia", determines whether to use AUD or IDR for revenue and expense.
        tolerance: The tolerance level for matching random treatment revenue and expense.

        Returns:
        forecast_df: DataFrame with forecasted 'Period', 'Revenue', and 'Expenses'.
        """
        
        # Define which columns to use for revenue and expense based on the basis
        if basis == "Australia":
            revenue_col = 'total_price_AUD'
            expense_col = 'total_cost_AUD'
        elif basis == "Indonesia":
            revenue_col = 'total_price_IDR'
            expense_col = 'total_cost_IDR'
        else:
            raise ValueError("Basis must be 'Australia' or 'Indonesia'")
        
        # Calculate historical profit (Revenue - Expenses)
        historical_data_df['Profit'] = historical_data_df['Revenue'] - historical_data_df['Expense']
        
        # Extract the historical Revenue and Expenses
        historical_revenue = historical_data_df['Revenue'].values
        historical_expenses = historical_data_df['Expense'].values
        
        # Initialize forecast lists with the historical data for reference
        revenue_forecast = list(historical_revenue)
        expenses_forecast = list(historical_expenses)
        
        # Generate forecast for the desired number of periods
        for period in range(number_of_forecasted_periods):
            # For the first forecasted period, use the last MA_period rows of historical data
            if period == 0:
                revenue_ma = np.mean(revenue_forecast[-MA_period:])
                expenses_ma = np.mean(expenses_forecast[-MA_period:])
            else:
                # For subsequent periods, include the previous forecasted period in the moving average
                revenue_ma = np.mean(revenue_forecast[-MA_period:])
                expenses_ma = np.mean(expenses_forecast[-MA_period:])
            
            # Initialize accumulators for revenue and expense
            accumulated_revenue = 0
            accumulated_expense = 0
            selected_treatments = []

            # Select treatments based on the revenue and expense values with tolerance
            while not (revenue_ma * (1 - tolerance) <= accumulated_revenue <= revenue_ma * (1 + tolerance)
                    and expenses_ma * (1 - tolerance) <= accumulated_expense <= expenses_ma * (1 + tolerance)):
                
                # Randomly select a treatment from treatment_df using the basis-specific columns
                selected_treatment = treatment_df.sample(n=1).iloc[0]
                
                treatment_revenue = selected_treatment[revenue_col]
                treatment_expense = selected_treatment[expense_col]
                
                # Accumulate the selected treatment's revenue and expense
                accumulated_revenue += treatment_revenue
                accumulated_expense += treatment_expense
                
                selected_treatments.append({
                    'Treatment': selected_treatment['Treatment'],
                    'Revenue': treatment_revenue,
                    'Expense': treatment_expense
                })
            
            # Append the adjusted forecast values to the forecast list
            revenue_forecast.append(accumulated_revenue)
            expenses_forecast.append(accumulated_expense)
        
        # Create a forecast DataFrame, excluding the initial historical data
        forecast_periods = np.arange(start_month, number_of_forecasted_periods + start_month)
        forecast_df = pd.DataFrame({
            'Period': forecast_periods,
            'Revenue': np.round(revenue_forecast[-number_of_forecasted_periods:], 0),
            'Expense': np.round(expenses_forecast[-number_of_forecasted_periods:], 0)
        })
        
        return forecast_df



    def summary_table(self, sales_df, debt_repayment_df=None, indirect_expense_df=None, equipment_df=None, fitout_df=None, by='Product'):
        """
        Generates a summary DataFrame from sales_df and optionally debt_repayment_df by converting revenue and expenses into separate rows.

        Parameters:
        sales_df: DataFrame containing columns 'Period', 'Treatment', 'Revenue', 'Expense', 'Customer ID'.
        debt_repayment_df: Optional DataFrame containing columns 'Period', 'Expense' for debt repayment.
        indirect_expense_df: Optional DataFrame containing columns 'Period', 'Expense' for indirect expenses.

        Returns:
        fig: Plotly figure displaying the summary table.
        """
        
        # Initialize an empty list to collect rows for the summary table
        summary_data = []

        # Process sales_df
        for _, row in sales_df.iterrows():
            period = row['Period']
            treatment = row['Treatment']
            revenue = row['Revenue']
            expense = row['Expense']
            customer_id = row.get('Customer ID', None)  # Get Customer ID if available, else None
            
            # Add the row for Expense (Cash Out)
            summary_data.append({
                'Period': period,
                'From': "Clinic",
                'To': "Material and Salary",
                'Reason': f'COGS for {treatment}',
                'Customer ID': customer_id if expense >= 0 else None,  # Add Customer ID only for sales_df
                'Cash In': "",  # No cash inflow for expense
                'Cash Out': -expense,  # Negative value for expense

            })
            
            # Add the row for Revenue (Cash In)
            summary_data.append({
                'Period': period,
                'From': 'Customer',
                'To': 'Clinic',
                'Reason': f'Income for {treatment} Sales',
                'Customer ID': customer_id if revenue >= 0 else None,  # Add Customer ID only for sales_df
                'Cash In': revenue,  # Positive value for revenue
                'Cash Out': "",  # No cash outflow for revenue

            })

        # Process indirect_expenses only if provided
        if indirect_expense_df is not None:
            for _, row in indirect_expense_df.iterrows():
                period = row['Period']
                expense = row['Expense']
                
                # Add row for Indirect Expense (negative amount in Cash Out)
                summary_data.append({
                    'Period': period,
                    'From': 'Clinic',
                    'To': 'Indirect Expense',
                    'Reason': 'Operating and General Expenses',
                    'Cash In': 0,  # No cash inflow
                    'Cash Out': -expense,  # Negative value for expense
                    'Customer ID': None  # No Customer ID for indirect expenses
                })

        # Process debt_repayment_df only if provided
        if debt_repayment_df is not None:
            for _, row in debt_repayment_df.iterrows():
                period = row['Period']
                expense = int(row['Expense'])
                
                # Add row for Debt Repayment (negative amount in Cash Out)
                summary_data.append({
                    'Period': period,
                    'From': 'Clinic',
                    'To': 'Bank/Loan Provider',
                    'Reason': 'Debt Repayment',
                    'Cash In': 0,  # No cash inflow
                    'Cash Out': -expense,  # Negative value for expense
                    'Customer ID': None  # No Customer ID for debt repayments
                })
                
        if equipment_df is not None:
            for _, row in equipment_df.iterrows():
                period = row['Period']
                expense = row['Expense']
                equipment = row['Equipment']
                
                # Add row for Debt Repayment (negative amount in Cash Out)
                summary_data.append({
                    'Period': period,
                    'From': 'Clinic',
                    'To': 'Equipment Sellers',
                    'Reason': 'Buy New Equipment of ' + equipment,
                    'Cash In': 0,  # No cash inflow
                    'Cash Out': -expense,  # Negative value for expense
                    'Customer ID': None  # No Customer ID for debt repayments
                })
        
        if fitout_df is not None:
            for _, row in fitout_df.iterrows():
                period = row['Period']
                expense = row['Expense']
                
                # Add row for Debt Repayment (negative amount in Cash Out)
                summary_data.append({
                    'Period': period,
                    'From': 'Clinic',
                    'To': 'Fitout Provider',
                    'Reason': 'Fitout Activity',
                    'Cash In': 0,  # No cash inflow
                    'Cash Out': -expense,  # Negative value for expense
                    'Customer ID': None  # No Customer ID for debt repayments
                })
    

        
        if by == 'Product':
            # Convert the collected data into a DataFrame
            summary_df = pd.DataFrame(summary_data, columns=['Period', 'From', 'To', 'Reason', 'Cash In', 'Cash Out'])
            summary_df['Period'] = pd.to_datetime(summary_df['Period'], format='%d-%m-%Y')
            summary_df['Period'] = summary_df['Period'].dt.date 
            
            # Sort the DataFrame by Period
            summary_df = summary_df.sort_values(by='Period').reset_index(drop=True)
            
            # Define column widths
            column_widths = [3, 3, 3, 10, 2, 2]
            
        elif by == 'Customer':
            # Convert the collected data into a DataFrame
            summary_df = pd.DataFrame(summary_data, columns=['Period', 'From', 'To', 'Customer ID', 'Cash In', 'Cash Out'])
            
            summary_df['Period'] = pd.to_datetime(summary_df['Period'], format='%d-%m-%Y')
            summary_df['Period'] = summary_df['Period'].dt.date 
            
            # Sort the DataFrame by Period
            summary_df = summary_df.sort_values(by='Period').reset_index(drop=True)
            
            # Define column widths
            column_widths = [3, 3, 3, 4, 2, 2, 2]
        

        total_width = sum(column_widths)
        normalized_widths = [width / total_width for width in column_widths]
        
        # Create a Plotly table using summary_df
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_df.columns),
                        fill_color='grey',
                        align='left',
                        font=dict(color='white', size=15)),
            
            cells=dict(values=[summary_df[col] for col in summary_df.columns],
                    fill_color='lightgrey',
                    align='left'),
            columnwidth=normalized_widths  # Apply the normalized widths to the columns
        )])
        
        fig.update_layout(width=800, height=700)
        
        return fig
    
    
    def equipment_to_cashflow(self, df, period, start_year=2024):
        # Create a list to hold data for the output DataFrame
        cashflow_data = []
        
        df['Expected Lifetime'] = df['Expected Lifetime'] * 12
        df['Current Lifetime Usage'] = df['Current Lifetime Usage'] * 12
        df['Remaining Lifetime'] = df['Expected Lifetime'] - df['Current Lifetime Usage']

        for i in range(1, period + 1):
            # Filter equipment that needs to be repurchased at the current period
            equipment_due = df[df['Remaining Lifetime'] == i]
            
            if not equipment_due.empty:
                for _, row in equipment_due.iterrows():
                    cashflow_data.append({
                        'Period': i,
                        'Equipment': row['Equipment'],
                        'Expense': row['Price']
                    })
            else:
                # Append a row with Equipment as None and Expense as 0 when no equipment is due
                cashflow_data.append({
                    'Period': i,
                    'Equipment': 'None',
                    'Expense': 0
                })
                
        
        result = pd.DataFrame(cashflow_data)
        result['Period'] = result['Period'].apply(lambda period: self.generate_date_from_month(int(period), method='first_day', start_year=start_year))

        result['Revenue'] = 0  # Add a Revenue column with 0 values
        
        result['Period'] = pd.to_datetime(result['Period'], format='%d-%m-%Y')
        result['Period'] = result['Period'].dt.date
        result = result.sort_values(by='Period').reset_index(drop=True)
        
        
        
        return result


    def fitout_to_cashflow(self, fitout_value, last_fitout, period, start_year=2024):
        
        period_to_fitout = 10 # in years
        
        next_fitout_period = period_to_fitout - last_fitout # in years
        next_fitout_period_in_months = next_fitout_period * 12
        
        cashflow_data = []
        
        for i in range(1, period + 1):
            # Filter equipment that needs to be repurchased at the current period
            fitout_due = next_fitout_period_in_months
            
            if i == fitout_due:
   
                cashflow_data.append({
                    'Period': i,
                    'Expense': fitout_value
                })
            else:
                # Append a row with Equipment as None and Expense as 0 when no equipment is due
                cashflow_data.append({
                    'Period': i,
                    'Expense': 0
                })
                
        result = pd.DataFrame(cashflow_data)
        result['Period'] = result['Period'].apply(lambda period: self.generate_date_from_month(int(period), method='first_day', start_year=start_year))

        result['Revenue'] = 0  # Add a Revenue column with 0 values
        
        result['Period'] = pd.to_datetime(result['Period'], format='%d-%m-%Y')
        result['Period'] = result['Period'].dt.date
        result = result.sort_values(by='Period').reset_index(drop=True)
        
        
        return result
    
    
    def forecast_revenue_expenses_given_item_code(self, clinic_item_code, cleaned_item_code, forecast_period, target_cov, tolerance=0.1):
        # Ensure consistent formatting by zero-padding 'Code' to match 'item_number'
        clinic_item_code['Code'] = clinic_item_code['Code'].apply(lambda x: str(x).zfill(3))
        cleaned_item_code['item_number'] = cleaned_item_code['item_number'].astype(str).str.zfill(3)
        cleaned_item_code['salary_per_code'] = cleaned_item_code['Total Salary'] / cleaned_item_code['Total Demand']
        
        # Merge clinic_item_code with cleaned_item_code based on Code and item_number
        merged_df = pd.merge(
            clinic_item_code,
            cleaned_item_code,
            left_on="Code",
            right_on="item_number",
            how="inner"
        )
        
        # Extract necessary columns
        merged_df = merged_df[['Code', 'Total Demand', 'price AUD', 'cost_material AUD', 'salary_per_code']]
        
        # Scale Total Demand to match the forecast_period
        merged_df['Demand per Period'] = (merged_df['Total Demand'] / 12) * forecast_period

        # Initialize cashflow DataFrame
        periods = list(range(1, forecast_period + 1))
        cashflow_df = pd.DataFrame({'Period': periods})
        cashflow_df['Revenue'] = 0.0
        cashflow_df['Expense'] = 0.0
        
        # Generate random allocations and control CoV
        def calculate_cov(values):
            """Calculate coefficient of variation."""
            mean = np.mean(values)
            std_dev = np.std(values)
            return std_dev / mean if mean != 0 else 0
        
        valid_distribution = False
        
        while not valid_distribution:
            # Reset Revenue and Expense
            cashflow_df['Revenue'] = 0.0
            cashflow_df['Expense'] = 0.0
            
            for _, row in merged_df.iterrows():
                # Randomly allocate demand across forecast_period
                random_allocation = np.random.dirichlet(np.ones(forecast_period)) * row['Demand per Period']
                
                # Calculate revenue and expense for the allocated demand
                revenue = random_allocation * row['price AUD']
                expense_lab_material = random_allocation * row['cost_material AUD']
                expense_salary = random_allocation * row['salary_per_code']
                
                expense = expense_lab_material + expense_salary
                
                # Add to cashflow DataFrame
                cashflow_df['Revenue'] += revenue
                cashflow_df['Expense'] += expense
            
            # Calculate Revenue - Expense
            cashflow_df['Profit'] = cashflow_df['Revenue'] - cashflow_df['Expense']
            
            # Calculate CoV of Profit
            cov = calculate_cov(cashflow_df['Profit'])
            
            # Check if CoV is within the target range
            if target_cov * (1 - tolerance) <= cov <= target_cov * (1 + tolerance):
                valid_distribution = True
        
        # Drop the Profit column before returning the DataFrame
        cashflow_df = cashflow_df[['Period', 'Revenue', 'Expense']]
        
        #round cashflow_df
        cashflow_df['Revenue'] = np.round(cashflow_df['Revenue'], 0)
        cashflow_df['Expense'] = np.round(cashflow_df['Expense'], 0)

        
        return cashflow_df
    
    
    def generate_monthly_cashflow_given_item_code(self, clinic_item_code, cleaned_item_code, forecast_period, target_cov, start_year=2024, start_month=1):
        """
        Optimized generation of a monthly cashflow DataFrame with controlled CoV for revenue.
        """
        import pandas as pd
        import numpy as np
        import random
        import calendar
        from datetime import date
        
        start_year += 1 # somehow need to be added by 1

        clinic_item_code['Code'] = clinic_item_code['Code'].apply(lambda x: str(x).zfill(3))
        cleaned_item_code['item_number'] = cleaned_item_code['item_number'].astype(str).str.zfill(3)

        clinic_item_code['price AUD'] = clinic_item_code['Code'].map(cleaned_item_code.set_index('item_number')['price AUD'])
        clinic_item_code['cost_material AUD'] = clinic_item_code['Code'].map(cleaned_item_code.set_index('item_number')['cost_material AUD'])
        
        clinic_item_code['salary_per_code'] = clinic_item_code['Total Salary'] / clinic_item_code['Total Demand']
        
        clinic_item_code['Total Demand'] = clinic_item_code['Total Demand'] / 12 * forecast_period
        clinic_item_code['Total Revenue'] = clinic_item_code['Total Revenue'] / 12 * forecast_period
        clinic_item_code['Total Material Cost'] = clinic_item_code['Total Material Cost'] / 12 * forecast_period
        clinic_item_code['Total Duration'] = clinic_item_code['Total Duration'] / 12 * forecast_period
        clinic_item_code['Total Salary'] = clinic_item_code['Total Salary'] / 12 * forecast_period


        # Calculate total revenue and expense
        total_revenue = (clinic_item_code['Total Demand'] * clinic_item_code['price AUD']).sum() 
        total_expense = (clinic_item_code['Total Demand'] * clinic_item_code['cost_material AUD']).sum() 

        # Initialize monthly cashflow
        monthly_cashflow = pd.DataFrame({
            'Period': list(range(1, forecast_period + 1)),
            'Revenue': np.zeros(forecast_period),
            'Expense': np.zeros(forecast_period),
            'Profit': np.zeros(forecast_period)
        })

        # Generate Revenue and Expense with controlled CoV for Revenue
        mean_revenue = total_revenue / (forecast_period/12) / 12 # as the total_revenue is for a year
        mean_expense = total_expense / (forecast_period/12) / 12 # as the total_expense is for a year

        std_revenue = target_cov * mean_revenue

        # Generate revenue distribution with target CoV
        revenue_dist = np.random.normal(loc=mean_revenue, scale=std_revenue, size=forecast_period)
        revenue_dist = np.clip(revenue_dist, 0, None)  # Ensure no negative revenues

        # Scale Revenue to match total
        revenue_dist *= total_revenue / revenue_dist.sum()

        # Adjust Expense and Profit
        expense_dist = (revenue_dist / total_revenue) * total_expense
        profit_dist = revenue_dist - expense_dist

        # Populate monthly cashflow
        monthly_cashflow['Revenue'] = revenue_dist
        monthly_cashflow['Expense'] = expense_dist
        monthly_cashflow['Profit'] = profit_dist

        # Populate Codes by Period based on Revenue
        def populate_item_codes_by_period(clinic_item_code, monthly_cashflow):
            """
            Populate item codes into periods based on target revenues in the monthly cashflow.
            """
            # Initialize a DataFrame for the final output
            populated_data = []

            # Create a working copy of clinic_item_code with a Remaining Demand column
            clinic_item_code = clinic_item_code.copy()
            clinic_item_code['Remaining Demand'] = clinic_item_code['Total Demand']

            for _, period_row in monthly_cashflow.iterrows():
                period = period_row['Period']
                target_revenue = period_row['Revenue']
                accumulated_revenue = 0

                while accumulated_revenue < target_revenue and clinic_item_code['Remaining Demand'].sum() > 0:
                    # Randomly select an item code
                    available_items = clinic_item_code[clinic_item_code['Remaining Demand'] > 0]
                    sampled_row = available_items.sample(n=1).iloc[0]

                    # Calculate the sampled revenue
                    sampled_revenue = sampled_row['price AUD']
                    sampled_expense_lab_material = sampled_row['cost_material AUD']
                    sampled_expense_salary = sampled_row['salary_per_code']
                    sampled_expense = sampled_expense_lab_material + sampled_expense_salary
                    sampled_profit = sampled_revenue - sampled_expense

                    # Deduct one unit of demand from the sampled item code
                    clinic_item_code.loc[sampled_row.name, 'Remaining Demand'] -= 1

                    # Append to the populated data
                    populated_data.append({
                        'Period': period,
                        'Code': sampled_row['Code'],
                        'Revenue': sampled_revenue,
                        'Expense': sampled_expense,
                        'Profit': sampled_profit
                    })

                    accumulated_revenue += sampled_revenue

            # Handle remaining item codes in the last period
            last_period = monthly_cashflow['Period'].max()
            remaining_items = clinic_item_code[clinic_item_code['Remaining Demand'] > 0]

            for _, remaining_row in remaining_items.iterrows():
                while remaining_row['Remaining Demand'] > 0:
                    remaining_row['Remaining Demand'] -= 1
                    populated_data.append({
                        'Period': last_period,
                        'Code': remaining_row['Code'],
                        'Revenue': remaining_row['price AUD'],
                        'Expense': remaining_row['cost_material AUD'],
                        'Profit': remaining_row['price AUD'] - remaining_row['cost_material AUD']
                    })

            # Convert to DataFrame
            populated_df = pd.DataFrame(populated_data)

            return populated_df

        populated_item_codes_df = populate_item_codes_by_period(clinic_item_code, monthly_cashflow)

        # Convert Period to actual dates starting from January 2024
        def convert_period_to_date_with_calendar_logic(monthly_cashflow_df, start_year, start_month):
            """
            Convert numerical periods into specific dates based on a calendar starting from start_month of start_year.
            """
            def random_date_from_period(period):
                # Calculate the year and month based on the starting month and year
                total_months = (start_year - 1) * 12 + start_month - 1 + (period - 1)
                year = total_months // 12
                month = total_months % 12 + 1
                days_in_month = calendar.monthrange(year, month)[1]
                random_day = random.randint(1, days_in_month)
                return date(year, month, random_day)

            monthly_cashflow_df['Period'] = monthly_cashflow_df['Period'].astype(int)
            monthly_cashflow_df['Date'] = monthly_cashflow_df['Period'].apply(random_date_from_period)
            monthly_cashflow_df['Period'] = monthly_cashflow_df['Date']

            monthly_cashflow_df.drop(columns=['Date'], inplace=True)

            return monthly_cashflow_df

        # Final conversion to date
        converted_cashflow_df = convert_period_to_date_with_calendar_logic(populated_item_codes_df, start_year, start_month)

        # Sorting and formatting
        converted_cashflow_df['Period'] = pd.to_datetime(converted_cashflow_df['Period'], format='%d-%m-%Y')
        converted_cashflow_df['Period'] = converted_cashflow_df['Period'].dt.date
        converted_cashflow_df = converted_cashflow_df.sort_values(by='Period').reset_index(drop=True)

        return converted_cashflow_df
    
    
    def add_hourly_period(self, converted_cashflow_df):
        """
        Add a new column 'Hourly_Period' to the cashflow DataFrame by assigning random times
        to each date in the 'Period' column. The times are in half-hour increments and range
        from 8:00 AM to 12:00 PM and 1:00 PM to 4:00 PM.

        Parameters:
        - converted_cashflow_df: DataFrame with a 'Period' column containing date values.

        Returns:
        - DataFrame with an added 'Hourly_Period' column.
        """
        # Define valid time slots
        morning_slots = [timedelta(hours=8 + i // 2, minutes=(i % 2) * 30) for i in range(8)]
        afternoon_slots = [timedelta(hours=13 + i // 2, minutes=(i % 2) * 30) for i in range(8)]
        valid_time_slots = morning_slots + afternoon_slots

        # Function to generate a random hourly period
        def generate_random_hourly_period(period_date):
            random_slot = random.choice(valid_time_slots)
            return datetime.combine(period_date, datetime.min.time()) + random_slot

        # Apply the function to generate 'Hourly_Period'
        converted_cashflow_df['Hourly_Period'] = converted_cashflow_df['Period'].apply(generate_random_hourly_period)

        return converted_cashflow_df
