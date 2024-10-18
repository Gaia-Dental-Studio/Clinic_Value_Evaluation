import pandas as pd
import numpy as np
import plotly.express as px



class ModelForecastPerformance:
    def __init__(self, company_variables):
        
        self.ebit = company_variables['EBIT']
        self.net_sales_growth = company_variables['Net Sales Growth']
        self.relative_variability_net_sales = company_variables['Relative Variation of Net Sales']





    def forecast_ebit_flow(self, number_of_forecasted_periods):
        """
        Generates a DataFrame of forecasted EBIT values.
        
        Parameters:
        initial_EBIT: The EBIT at period 0 (not displayed).
        ebit_growth_monthly: The average monthly growth rate of EBIT.
        number_of_forecasted_periods: Number of forecast periods (months).
        relative_variation: Desired coefficient of variation (CV) of the forecasted EBIT values.
        
        Returns:
        DataFrame with two columns: 'Period' (1 to N) and 'EBIT'.
        """
        
        initial_EBIT = self.ebit / 12
        ebit_growth_monthly = self.net_sales_growth / 12
        relative_variation = self.relative_variability_net_sales
        
        # Create an array for the period
        periods = np.arange(1, number_of_forecasted_periods + 1)
        
        # Calculate the deterministic growth component (trend)
        trend = initial_EBIT * (1 + ebit_growth_monthly) ** periods
        
        # Introduce random variation based on the relative variation (CV)
        # Coefficient of variation is std/mean, so we introduce noise with appropriate std
        random_variation = np.random.normal(loc=0, scale=relative_variation, size=number_of_forecasted_periods)
        
        # Combine the trend with the variation, ensuring positive EBIT values
        ebit_values = trend * (1 + random_variation)
        
        # Create a DataFrame for the result
        forecast_df = pd.DataFrame({
            'Period': periods,
            'EBIT': np.round(ebit_values, 0)  # Round EBIT values to 2 decimal places
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
