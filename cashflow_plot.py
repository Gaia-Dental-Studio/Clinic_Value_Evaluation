import pandas as pd
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict


class ModelCashflow:
    def __init__(self):
        # Initialize an empty dictionary to hold dataframes for different companies
        self.collection_df = {}

    def add_company_data(self, company_name, df):
        """
        Adds a new company's cashflow dataframe to the collection.
        :param company_name: Name of the company (string).
        :param df: Dataframe containing Period, Revenue, Expense columns.
        """
        # Check if the dataframe has the correct columns
        if not {'Period', 'Revenue', 'Expense'}.issubset(df.columns):
            raise ValueError("Dataframe must contain 'Period', 'Revenue', and 'Expense' columns.")
        
        # Ensure Period is treated as integer values for plotting
        # df['Period'] = df['Period'].astype(int)
        
        # Add company data to collection
        self.collection_df[company_name] = df
        
    def combine_and_average(self):
        """
        Combines all DataFrames in collection_df by summing up their Revenue and Expense columns.
        Returns the average total revenue and average total expense across all periods.
        """
        if not self.collection_df:
            raise ValueError("No company data has been added.")
        
        # Combine all DataFrames on Period by summing Revenues and Expenses
        combined_df = pd.DataFrame(columns=['Period', 'Total Revenue', 'Total Expense'])
        
        for company, df in self.collection_df.items():
            if combined_df.empty:
                combined_df = df.copy()
                combined_df.rename(columns={'Revenue': 'Total Revenue', 'Expense': 'Total Expense'}, inplace=True)
            else:
                # Merge on Period and sum Revenue and Expense
                combined_df = pd.merge(combined_df, df, on='Period', how='outer', suffixes=('', '_drop'))
                combined_df['Total Revenue'] += combined_df['Revenue'].fillna(0)
                combined_df['Total Expense'] += combined_df['Expense'].fillna(0)
                
                # Drop the additional columns created by the merge
                combined_df.drop(columns=['Revenue', 'Expense'], inplace=True)
        
        # Sort by Period
        combined_df.sort_values(by='Period', inplace=True)
        
        # Calculate total revenue and total expense
        total_revenue_sum = combined_df['Total Revenue'].sum()
        total_expense_sum = combined_df['Total Expense'].sum()
        
        # Calculate the average total revenue and total expense
        period_count = combined_df['Period'].nunique()
        avg_total_revenue = total_revenue_sum / period_count
        avg_total_expense = total_expense_sum / period_count

        return combined_df, avg_total_revenue, avg_total_expense
        
    def remove_all_companies(self):
        """
        Removes all companies' data from the collection.
        """
        self.collection_df = {}





    def merge_companies_or_categories(self, companies_dictionary, by='companies'):
        
        new_categories_dictionary = defaultdict(list)
        merged_categories_dictionary = {}
        
        if by == 'companies':
            # Merge by companies (original functionality)
            new_companies_dictionary = {}

            for clinic_name, dataframes in companies_dictionary.items():
                # Initialize an empty list to hold all DataFrames for this clinic_name
                merged_df_list = []

                # Iterate through each dataframe in the inner dictionary
                for dataframe_key, dataframe in dataframes.items():
                    # Ensure the 'Period' column is in datetime format
                    dataframe['Period'] = pd.to_datetime(dataframe['Period'])
                    # Append the dataframe to the list
                    merged_df_list.append(dataframe)

                # Combine all dataframes for this clinic_name on 'Period'
                merged_df = pd.concat(merged_df_list, axis=0).sort_values('Period').reset_index(drop=True)
                merged_df = merged_df.groupby('Period', as_index=False).agg({
                    'Revenue': 'sum',
                    'Expense': 'sum',
                    'Quarter': 'mean'
                })

                # Add a 'Profit' column
                merged_df['Profit'] = merged_df['Revenue'] - merged_df['Expense']

                # Drop rows where both Revenue and Expense are zero
                merged_df = merged_df[(merged_df['Revenue'] != 0) | (merged_df['Expense'] != 0)]

                # Store merged DataFrame by clinic_name
                new_companies_dictionary[clinic_name] = merged_df

   

        elif by == 'categories':
            # Merge by categories (new functionality)
            new_categories_dictionary = defaultdict(list)

            # Collect all dataframes under their respective category
            for clinic_name, dataframes in companies_dictionary.items():
                for dataframe_key, dataframe in dataframes.items():
                    # Ensure the 'Period' column is in datetime format
                    dataframe['Period'] = pd.to_datetime(dataframe['Period'])
                    # Append the dataframe to the category list
                    new_categories_dictionary[dataframe_key].append(dataframe)

            # Merge all dataframes within each category
            merged_categories_dictionary = {}
            for category, dataframes in new_categories_dictionary.items():
                merged_df = pd.concat(dataframes, axis=0).sort_values('Period').reset_index(drop=True)
                merged_df = merged_df.groupby('Period', as_index=False).agg({
                    'Revenue': 'sum',
                    'Expense': 'sum',
                    'Quarter': 'mean'
                })

                # Add a 'Profit' column
                merged_df['Profit'] = merged_df['Revenue'] - merged_df['Expense']

                # Drop rows where both Revenue and Expense are zero
                merged_df = merged_df[(merged_df['Revenue'] != 0) | (merged_df['Expense'] != 0)]

                # Store the merged DataFrame by category
                merged_categories_dictionary[category] = merged_df

        
        final_dictionaries = new_companies_dictionary if by == 'companies' else merged_categories_dictionary
        

        self.remove_all_companies()
        
        for company_name, df in final_dictionaries.items():
            self.add_company_data(company_name, df)
            
        
        return final_dictionaries



    def cashflow_plot(self, number_of_days, granularity='daily', start_date='2024-01-01'):
        """
        Generates an interactive Plotly cashflow plot using Period as the x-axis.
        Allows granularity to be set to 'daily', 'weekly', or 'monthly'.
        """
        # Create a Plotly figure
        fig = go.Figure()

        # Starting date
        start_date = pd.Timestamp(start_date)

        # Generate date_array based on the input number_of_days
        date_array = pd.date_range(start=start_date, periods=number_of_days, freq='D')
        date_series = pd.Series(date_array, name="Period")
        elongated_df = pd.DataFrame({'Period': date_series})

        # Variable to accumulate the total revenue and expense
        accumulated_revenue = None
        accumulated_expense = None

        # Define a function to generate the Period label based on the granularity
        def get_period_label(date, granularity):
            if granularity == 'weekly':
                return f"W{date.isocalendar().week - date.replace(day=1).isocalendar().week + 1}-{date.strftime('%b')}-{date.strftime('%y')}"
            elif granularity == 'monthly':
                return date.strftime('%b-%y')
            elif granularity == 'quarterly':  # New case for quarterly
                quarter = (date.month - 1) // 3 + 1
                return f"{date.year}-Q{quarter}"
            else:  # 'daily'
                return date

        # Loop through each company in the collection and process data
        for company_name, df in self.collection_df.items():
            df['Period'] = pd.to_datetime(df['Period'])
            
            # df = df.groupby('Period', as_index=False).agg({
            #     'Revenue': 'sum',
            #     'Expense': 'sum',
            #     'Order': 'mean'
            # })

            
            # Align company DataFrame with the elongated date range
            df = pd.merge(elongated_df, df, on='Period', how='left')
            df['Revenue'].fillna(0, inplace=True)
            df['Expense'].fillna(0, inplace=True)

            # Create a new Period label based on the granularity
            df['GranularPeriod'] = df['Period'].apply(lambda x: get_period_label(x, granularity))
            df['Order'] = df['Period'].rank(method='dense').astype(int)

            # Aggregate the data based on the new GranularPeriod
            if granularity != 'daily':
                df = df.groupby('GranularPeriod', as_index=False).agg({
                    'Revenue': 'sum',
                    'Expense': 'sum',
                    'Order': 'mean'
                })
            else:
                # df['GranularPeriod'] = df['Period']  # Keep original date for daily
                df = df.groupby('GranularPeriod', as_index=False).agg({
                'Revenue': 'sum',
                'Expense': 'sum',
                'Order': 'mean'
            })


            df = df.sort_values(by='Order')

            # Add bars for expenses for each company
            fig.add_trace(go.Bar(
                x=df['GranularPeriod'], 
                y=-df['Expense'],  # Expenses are negative
                name='Expense',
                marker_color='#f04343',
                legendgroup=company_name,
                showlegend=False,
                customdata=np.stack((df['Revenue'], df['Expense']), axis=-1),
                hovertemplate=(
                    f'<b>{company_name}</b><br>' +
                    'Expense: %{customdata[1]:,.0f}<br>'
                )
            ))

            # Add bars for revenue for each company
            fig.add_trace(go.Bar(
                x=df['GranularPeriod'], 
                y=df['Revenue'],
                name='Revenue',
                marker_color='#337dd6',
                legendgroup=company_name,
                showlegend=False,
                customdata=np.stack((df['Revenue'], df['Expense']), axis=-1),
                hovertemplate=(
                    f'<b>{company_name}</b><br>' +
                    'Revenue: %{customdata[0]:,.0f}<br>'
                )
            ))

            # Add invisible scatter trace for company name in the legend
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color='grey'),
                name=company_name,
                legendgroup=company_name,
                showlegend=True
            ))

            # Accumulate the revenue and expense for cumulative calculation
            if accumulated_revenue is None:
                accumulated_revenue = df['Revenue'].copy()
                accumulated_expense = df['Expense'].copy()
            else:
                accumulated_revenue += df['Revenue']
                accumulated_expense += df['Expense']

        # Calculate net and cumulative cashflow
        net_cashflow = accumulated_revenue - accumulated_expense
        cumulative_cashflow = net_cashflow.cumsum()

        # Add a line for cumulative cashflow
        fig.add_trace(go.Scatter(
            x=df['GranularPeriod'],
            y=cumulative_cashflow,
            mode='lines+markers',
            name='Cumulative Cashflow',
            marker_color='black',
            line=dict(dash='solid'),
            hovertemplate=(
                '<b>Cumulative Cashflow</b><br>' +
                'Period: %{x}<br>' +
                'Cumulative Cashflow: %{y:,.0f}<extra></extra>'
            )
        ))

        # Calculate the mean values for the reference lines
        mean_revenue = accumulated_revenue.mean()
        mean_expense = accumulated_expense.mean()
        max_revenue = mean_revenue * 1.3  # 30% deviation threshold
        max_expense = -mean_expense * 1.3  # 30% deviation threshold (negative)

        # Add horizontal reference lines for max allowable revenue and expense
        fig.add_hline(
            y=max_revenue,
            line=dict(color='blue', dash='dash'),
            name='Max Allowable Revenue (30% Tolerance)',
            annotation_text='Max Revenue (30% Tolerance)',
            annotation_position="top right"
        )
        fig.add_hline(
            y=max_expense,
            line=dict(color='red', dash='dash'),
            name='Max Allowable Expense (30% Tolerance)',
            annotation_text='Max Expense (30% Tolerance)',
            annotation_position="bottom right"
        )

        # Customize layout
        fig.update_layout(
            title=f'Cashflow ({granularity.capitalize()})',
            yaxis_title='Amount',
            xaxis_title='Period',
            hovermode='x',
            bargap=0.2,
            plot_bgcolor='white',
            barmode='relative'
        )

        # Return the interactive chart
        return fig


    
    def create_profit_comparison_matrix(self, scenario_metrics_dict):
        # Step 1: Calculate the profit for each scenario
        scenario_profits = {}
        for key, value in scenario_metrics_dict.items():
            avg_revenue, avg_expense = value
            profit = avg_revenue - avg_expense
            scenario_profits[key] = profit
        
        # Step 2: Create the comparison matrix
        scenario_labels = [f'Scenario {key}' for key in scenario_metrics_dict.keys()]
        comparison_matrix = pd.DataFrame(index=scenario_labels, columns=scenario_labels)
        
        # Step 3: Populate the matrix with the percentage difference in profit
        for row_key, row_profit in scenario_profits.items():
            for col_key, col_profit in scenario_profits.items():
                # Calculate percentage difference: ((row_profit - col_profit) / col_profit) * 100
                if col_profit != 0:
                    percentage_diff = ((row_profit - col_profit) / col_profit) * 100
                    percentage_diff = round(percentage_diff, 0)  # Round to 0 decimal places
                    percentage_diff = f"{int(percentage_diff)}%"  # Convert to percentage format
                else:
                    percentage_diff = "N/A"  # Handle division by zero if col_profit is 0
                # Set the percentage difference in the matrix
                comparison_matrix.loc[f'Scenario {row_key}', f'Scenario {col_key}'] = percentage_diff
        
        return comparison_matrix
    
    
    
    def cashflow_plot_detailed(self, number_of_intervals, granularity='half-hourly', start_date='2024-01-01 00:00:00'):
        """
        Generates an interactive Plotly cashflow plot using detailed granularity: half-hourly or hourly.
        Allows granularity to be set to 'half-hourly' or 'hourly'.
        """
        # Create a Plotly figure
        fig = go.Figure()

        # Starting date
        start_date = pd.Timestamp(start_date)

        # Generate date_array based on the input number_of_intervals
        freq = '30T' if granularity == 'half-hourly' else 'H'  # Frequency: '30T' for half-hourly, 'H' for hourly
        date_array = pd.date_range(start=start_date, periods=number_of_intervals, freq=freq)
        date_series = pd.Series(date_array, name="Hourly_Period")
        elongated_df = pd.DataFrame({'Hourly_Period': date_series})

        # Variable to accumulate the total revenue and expense
        accumulated_revenue = None
        accumulated_expense = None

        # Define a function to generate the Period label based on the granularity
        def get_period_label(date, granularity):
            if granularity == 'half-hourly':
                return date.strftime('%Y-%m-%d %H:%M')
            elif granularity == 'hourly':
                return date.strftime('%Y-%m-%d %H:00')
            else:
                raise ValueError("Unsupported granularity. Choose 'half-hourly' or 'hourly'.")

        # Loop through each company in the collection and process data
        for company_name, df in self.collection_df.items():
            df['Hourly_Period'] = pd.to_datetime(df['Hourly_Period'])

            # Align company DataFrame with the elongated date range
            df = pd.merge(elongated_df, df, on='Hourly_Period', how='left')
            df['Revenue'].fillna(0, inplace=True)
            df['Expense'].fillna(0, inplace=True)

            # Create a new Period label based on the granularity
            df['GranularPeriod'] = df['Hourly_Period'].apply(lambda x: get_period_label(x, granularity))
            df['Order'] = df['Hourly_Period'].rank(method='dense').astype(int)

            # Aggregate the data based on the new GranularPeriod
            df = df.groupby('GranularPeriod', as_index=False).agg({
                'Revenue': 'sum',
                'Expense': 'sum',
                'Order': 'mean'
            })
            df = df.sort_values(by='Order')

            # Add bars for expenses for each company
            fig.add_trace(go.Bar(
                x=df['GranularPeriod'], 
                y=-df['Expense'],  # Expenses are negative
                name='Expense',
                marker_color='#f04343',
                legendgroup=company_name,
                showlegend=False,
                customdata=np.stack((df['Revenue'], df['Expense']), axis=-1),
                hovertemplate=(
                    f'<b>{company_name}</b><br>' +
                    'Expense: %{customdata[1]:,.0f}<br>'
                )
            ))

            # Add bars for revenue for each company
            fig.add_trace(go.Bar(
                x=df['GranularPeriod'], 
                y=df['Revenue'],
                name='Revenue',
                marker_color='#337dd6',
                legendgroup=company_name,
                showlegend=False,
                customdata=np.stack((df['Revenue'], df['Expense']), axis=-1),
                hovertemplate=(
                    f'<b>{company_name}</b><br>' +
                    'Revenue: %{customdata[0]:,.0f}<br>'
                )
            ))

            # Add invisible scatter trace for company name in the legend
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(color='grey'),
                name=company_name,
                legendgroup=company_name,
                showlegend=True
            ))

            # Accumulate the revenue and expense for cumulative calculation
            if accumulated_revenue is None:
                accumulated_revenue = df['Revenue'].copy()
                accumulated_expense = df['Expense'].copy()
            else:
                accumulated_revenue += df['Revenue']
                accumulated_expense += df['Expense']

        # Calculate net and cumulative cashflow
        net_cashflow = accumulated_revenue - accumulated_expense
        cumulative_cashflow = net_cashflow.cumsum()

        # Add a line for cumulative cashflow
        fig.add_trace(go.Scatter(
            x=df['GranularPeriod'],
            y=cumulative_cashflow,
            mode='lines+markers',
            name='Cumulative Cashflow',
            marker_color='black',
            line=dict(dash='solid'),
            hovertemplate=(
                '<b>Cumulative Cashflow</b><br>' +
                'Period: %{x}<br>' +
                'Cumulative Cashflow: %{y:,.0f}<extra></extra>'
            )
        ))

        # Calculate the mean values for the reference lines
        mean_revenue = accumulated_revenue.mean()
        mean_expense = accumulated_expense.mean()
        max_revenue = mean_revenue * 1.3  # 30% deviation threshold
        max_expense = -mean_expense * 1.3  # 30% deviation threshold (negative)

        # Add horizontal reference lines for max allowable revenue and expense
        fig.add_hline(
            y=max_revenue,
            line=dict(color='blue', dash='dash'),
            name='Max Allowable Revenue (30% Tolerance)',
            annotation_text='Max Revenue (30% Tolerance)',
            annotation_position="top right"
        )
        fig.add_hline(
            y=max_expense,
            line=dict(color='red', dash='dash'),
            name='Max Allowable Expense (30% Tolerance)',
            annotation_text='Max Expense (30% Tolerance)',
            annotation_position="bottom right"
        )

        # Customize layout
        fig.update_layout(
            title=f'Cashflow ({granularity.capitalize()})',
            yaxis_title='Amount',
            xaxis_title='Period',
            hovermode='x',
            bargap=0.2,
            plot_bgcolor='white',
            barmode='relative'
        )

        # Return the interactive chart
        return fig
