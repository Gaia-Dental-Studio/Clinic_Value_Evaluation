import pandas as pd
import numpy as np

class ModelRecommendation:
    def __init__(self, transaction_data, item_code_details, recommendation_pairs, OHT_salary=35, specialist_salary=125):
        self.transaction_data = transaction_data
        self.converted_transaction_data = pd.DataFrame()
        self.item_code_details = item_code_details
        self.recommendation_pairs = recommendation_pairs
        self.OHT_salary = OHT_salary
        self.specialist_salary = specialist_salary

    def generate_converted_transaction_data(self, starting_conversion_rate, conversion_growth_increment, optimal_conversion_rate):
        # Extract month-year information from the Period column
        
        self.transaction_data['Period'] = pd.to_datetime(self.transaction_data['Period'], errors='coerce') 
        
        self.transaction_data['Month-Year'] = self.transaction_data['Period'].dt.to_period("M")
        
        # Sort transaction data by Period
        self.transaction_data.sort_values(by='Period', inplace=True)

        # Get unique months in order
        unique_months = self.transaction_data['Month-Year'].unique()

        # Initialize the conversion rate
        current_conversion_rate = starting_conversion_rate

        # Process transactions month-by-month
        for month in unique_months:
            # Filter rows belonging to the current month
            monthly_transactions = self.transaction_data[self.transaction_data['Month-Year'] == month]
            
            print(current_conversion_rate)

            for _, row in monthly_transactions.iterrows():
                # Apply the conversion rate logic
                if np.random.rand() > current_conversion_rate:
                    continue
                
                

                converted_row = row.copy()
                converted_row['Treatment'] = self.recommendation_pairs.loc[
                    self.recommendation_pairs['item_number'] == row['Treatment'], 'recommended_item'
                ].values[0]

                converted_row['Revenue'] = self.item_code_details.loc[
                    self.item_code_details['item_number'] == converted_row['Treatment'], 'price AUD'
                ].values[0]

                COGS_material = self.item_code_details.loc[
                    self.item_code_details['item_number'] == converted_row['Treatment'], 'cost_material AUD'
                ].values[0]

                duration = self.item_code_details.loc[
                    self.item_code_details['item_number'] == converted_row['Treatment'], 'duration'
                ].values[0]

                if self.item_code_details.loc[
                    self.item_code_details['item_number'] == converted_row['Treatment'], 'medical_officer_new'
                ].values[0] == 'OHT':
                    COGS_salary = duration * self.OHT_salary / 60
                else:
                    COGS_salary = duration * self.specialist_salary / 60

                converted_row['Expense'] = COGS_material + COGS_salary

                converted_row['Period'] = pd.to_datetime(converted_row['Period'], errors='coerce')
                converted_row['Period'] += pd.Timedelta(days=np.random.randint(7, 14))

                self.converted_transaction_data = pd.concat(
                    [self.converted_transaction_data, pd.DataFrame([converted_row])], ignore_index=True
                )

            # Increment conversion rate for the next month
            current_conversion_rate = min(current_conversion_rate + conversion_growth_increment, optimal_conversion_rate)

        return self.converted_transaction_data

    def group_by_period(self):
        # Return after grouping by month-year
        return self.converted_transaction_data.groupby(
            self.converted_transaction_data['Period'].dt.to_period("M")
        ).agg({'Revenue': 'sum', 'Expense': 'sum'}).reset_index()
