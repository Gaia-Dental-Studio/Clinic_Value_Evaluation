# Revised ModelFairCredit class with the same class and variable names
import pandas as pd
import numpy as np
import numpy_financial as npf

class ModelFairCredit:
    def __init__(self, transaction_data, minimum_allowable_price=500, applicant_ratio=0.5, period_term=10, monthly_interest=0.01, percentage_downpayment=0.3):
        self.transaction_data = transaction_data
        self.minimum_allowable_price = minimum_allowable_price
        self.applicant_ratio = applicant_ratio
        self.period_term = period_term
        self.monthly_interest = monthly_interest
        self.percentage_downpayment = percentage_downpayment
        
        self.aggregated_amortization_schedule = pd.DataFrame()
        
    def generate_amortization_schedule(self):
        adjusted_original_transaction = self.transaction_data.copy()
        adjusted_original_transaction['Adjusted'] = False  # Add a flag to track adjustments

        for _, row in self.transaction_data.iterrows():
            if row['Revenue'] < self.minimum_allowable_price:
                continue
            
            if np.random.rand() > self.applicant_ratio:
                continue

            amortization_schedule = pd.DataFrame()
            amortization_schedule['Period'] = pd.date_range(start=row['Period'], periods=self.period_term, freq='30D')
            
            price_as_principal = row['Revenue'] * (1 - self.percentage_downpayment)
            monthly_payment = npf.pmt(self.monthly_interest, self.period_term, -price_as_principal)
            
            amortization_schedule['Revenue'] = monthly_payment
            amortization_schedule['Expense'] = 0
            amortization_schedule['Customer ID'] = row['Customer ID']
            
            # Adjust the revenue for the specific row only
            mask = (
                (adjusted_original_transaction['Customer ID'] == row['Customer ID']) &
                (adjusted_original_transaction['Revenue'] == row['Revenue']) &
                (~adjusted_original_transaction['Adjusted'])  # Only unadjusted rows
            )
            if mask.any():
                adjusted_original_transaction.loc[mask, 'Revenue'] = row['Revenue'] * self.percentage_downpayment
                adjusted_original_transaction.loc[mask, 'Adjusted'] = True

            # Concatenate the amortization schedule to the aggregated schedule
            self.aggregated_amortization_schedule = pd.concat(
                [self.aggregated_amortization_schedule, amortization_schedule], ignore_index=True
            )
        
        # Drop the 'Adjusted' flag before returning
        adjusted_original_transaction = adjusted_original_transaction.drop(columns=['Adjusted'])
        return self.aggregated_amortization_schedule, adjusted_original_transaction
        
    def group_by_period(self):
        # Return after grouping by month-year
        return self.aggregated_amortization_schedule.groupby(self.aggregated_amortization_schedule['Period'].dt.to_period("M")).agg({'Revenue':'sum', 'Expense':'sum'}).reset_index()

