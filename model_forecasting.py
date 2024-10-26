import pandas as pd
import numpy as np
import plotly.express as px
import numpy_financial as npf
import plotly.graph_objects as go



class ModelForecastPerformance:
    def __init__(self, company_variables):
        
        self.ebit = company_variables['EBIT']
        self.net_sales_growth = company_variables['Net Sales Growth']
        self.relative_variability_net_sales = company_variables['Relative Variation of Net Sales']
        self.ebit_ratio = company_variables['EBIT Ratio']
        self.general_expense = -company_variables['General Expense']



    def forecast_indicect_cost(self, number_of_forecasted_periods):
        indirect_cost = self.general_expense / 12
        periods = np.arange(1, number_of_forecasted_periods + 1)
        
        # Create a DataFrame for the result
        forecast_df = pd.DataFrame({
            'Period': periods,
            'Revenue': 0,
            'Expense': np.round(indirect_cost, 0)  # Round Revenue values to 2 decimal places
        })
        
        return forecast_df
        


    def forecast_revenue_expenses(self, number_of_forecasted_periods):
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
            periods = np.arange(1, number_of_forecasted_periods + 1)
            
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
            print("New EBIT", total_ebit_after)
            print("Old EBIT", self.ebit)
            print("Minimum Growth", minimum_growth)
            print("Growth: ", growth)
            
            mean_ebit_values = np.mean(ebit_values)
            std_ebit_values = np.std(ebit_values)
            
            print("Coefficient of Variation: ", std_ebit_values / mean_ebit_values)
            
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


    
    def forecast_revenue_expenses_MA(self, historical_data_df, number_of_forecasted_periods, MA_period):
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
        forecast_periods = np.arange(1, number_of_forecasted_periods + 1)
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
    
    def loan_amortization_schedule(self, principal, annual_interest_rate, loan_term_years):
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
    


    def generate_forecast_treatment_df_by_profit(self, treatment_df, cashflow_df, basis="Australia", tolerance=0.1, number_of_unique_patient=1000):
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
                    'Period': period,
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
        
        return forecast_treatment_df






    def forecast_revenue_expenses_MA_by_profit(self, historical_data_df, number_of_forecasted_periods, MA_period, treatment_df, basis="Australia", tolerance=0.1):
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
        forecast_periods = np.arange(1, number_of_forecasted_periods + 1)
        forecast_df = pd.DataFrame({
            'Period': forecast_periods,
            'Revenue': np.round(revenue_forecast[-number_of_forecasted_periods:], 0),
            'Expense': np.round(expenses_forecast[-number_of_forecasted_periods:], 0)
        })
        
        return forecast_df



    def summary_table(self, sales_df, debt_repayment_df=None, indirect_expense_df=None, by='Product'):
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
                expense = row['Expense']
                
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
                    
            
            if by == 'Product':
                # Convert the collected data into a DataFrame
                summary_df = pd.DataFrame(summary_data, columns=['Period', 'From', 'To', 'Reason', 'Cash In', 'Cash Out'])
                
                # Sort the DataFrame by Period
                summary_df = summary_df.sort_values(by='Period').reset_index(drop=True)
                
                # Define column widths
                column_widths = [2, 3, 3, 11, 2, 2]
                
            elif by == 'Customer':
                # Convert the collected data into a DataFrame
                summary_df = pd.DataFrame(summary_data, columns=['Period', 'From', 'To', 'Customer ID', 'Cash In', 'Cash Out'])
                
                # Sort the DataFrame by Period
                summary_df = summary_df.sort_values(by='Period').reset_index(drop=True)
                
                # Define column widths
                column_widths = [2, 3, 3, 4, 2, 2, 2]
            
            
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

