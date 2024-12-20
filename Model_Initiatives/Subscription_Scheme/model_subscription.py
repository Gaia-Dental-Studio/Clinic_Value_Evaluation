import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import random

class ModelSubscription:
    def __init__(self, transaction_df, subscription_fee, conversion_ratio_monthly, days_until_inactive=90, churn_probability=0.1, start_date=None, end_date=None):
        self.transaction_df = transaction_df
        self.subscription_fee = subscription_fee
        self.conversion_ratio_monthly = conversion_ratio_monthly
        self.days_until_inactive = days_until_inactive
        self.churn_probability = churn_probability
        self.start_date = pd.to_datetime(start_date if start_date else transaction_df['Period'].min())
        self.end_date = pd.to_datetime(end_date if end_date else transaction_df['Period'].max())
        self.transformed_df = None
        self.patient_pool = {}

    def initialize_patient_pool(self):
        # Initialize the patient pool with non-converted patients
        self.transaction_df['Period'] = pd.to_datetime(self.transaction_df['Period'])
        for customer_id, group in self.transaction_df.groupby('Customer ID'):
            first_transaction = group['Period'].min()
            self.patient_pool[customer_id] = {
                'status': 'non-converted',
                'last_seen': first_transaction
            }

    def update_patient_pool(self, current_month):
        # Update the patient pool based on inactivity and new patients
        active_customers = self.transaction_df[self.transaction_df['Period'].dt.to_period('M') == current_month]['Customer ID'].unique()
        new_patients = []

        for customer_id in active_customers:
            if customer_id not in self.patient_pool:
                # Treat reappearing or new patients as new
                new_patients.append(customer_id)
                self.patient_pool[customer_id] = {
                    'status': 'non-converted',
                    'last_seen': current_month.start_time
                }
            else:
                # Update last_seen for existing patients
                self.patient_pool[customer_id]['last_seen'] = current_month.start_time

        # Mark patients inactive if they have been unseen for days_until_inactive
        for customer_id, details in self.patient_pool.items():
            if details['status'] == 'non-converted':
                if (current_month.start_time - details['last_seen']).days > self.days_until_inactive:
                    details['status'] = 'inactive'

        return new_patients

    def apply_monthly_conversion(self, current_month, new_patients):
        # Calculate the eligible pool for conversion
        eligible_non_converted = [cid for cid, details in self.patient_pool.items() if details['status'] == 'non-converted']
        eligible_pool = eligible_non_converted + new_patients

        num_to_convert = int(len(eligible_pool) * self.conversion_ratio_monthly)
        converting_patients = random.sample(eligible_pool, min(num_to_convert, len(eligible_pool)))

        for customer_id in converting_patients:
            self.patient_pool[customer_id]['status'] = 'converted'

        return converting_patients

    def apply_churn(self, current_month):
        # Simulate churn for converted patients
        if current_month == self.start_date.to_period('M'):
            return []  # No churn in the first month

        converted_patients = [cid for cid, details in self.patient_pool.items() if details['status'] == 'converted']
        churned_patients = [cid for cid in converted_patients if random.random() < self.churn_probability]

        for customer_id in churned_patients:
            self.patient_pool[customer_id]['status'] = 'non-converted'
            # Revert their transactions from current month onward
            customer_transactions = self.transformed_df[(self.transformed_df['Customer ID'] == customer_id) & 
                                                        (self.transformed_df['Period'] >= current_month.start_time)]
            original_transactions = self.transaction_df[self.transaction_df['Customer ID'] == customer_id]

            for index, transaction in customer_transactions.iterrows():
                period = transaction['Period']
                original_transaction = original_transactions[original_transactions['Period'] == period]
                if not original_transaction.empty:
                    # Transform back to the original transaction
                    self.transformed_df.loc[index, 'Revenue'] = original_transaction['Revenue'].values[0]
                    self.transformed_df.loc[index, 'Expense'] = original_transaction['Expense'].values[0]
                    self.transformed_df.loc[index, 'Treatment'] = original_transaction['Treatment'].values[0]

        return churned_patients

    def transform_to_subscription_scheme(self):
        self.initialize_patient_pool()
        self.transformed_df = self.transaction_df.copy()
        all_periods = pd.period_range(self.start_date, self.end_date, freq='M')

        for current_month in all_periods:
            new_patients = self.update_patient_pool(current_month)
            converting_patients = self.apply_monthly_conversion(current_month, new_patients)
            churned_patients = self.apply_churn(current_month)

            # Transform transactions for converting patients
            for customer_id in converting_patients:
                transactions = self.transformed_df[self.transformed_df['Customer ID'] == customer_id]
                self.transformed_df.loc[transactions.index, 'Revenue'] = 0

                # Add subscription rows
                subscription_rows = []
                subscription_date = current_month.start_time
                while subscription_date <= self.end_date:
                    subscription_rows.append({
                        'Period': subscription_date,
                        'Treatment': None,
                        'Revenue': self.subscription_fee,
                        'Expense': 0,
                        'Customer ID': customer_id
                    })
                    subscription_date += relativedelta(months=1)

                # Validate rows and convert to DataFrame
                subscription_rows_df = pd.DataFrame(subscription_rows)
                subscription_rows_df['Revenue'].fillna(self.subscription_fee, inplace=True)

                # Append subscription rows directly into the DataFrame
                self.transformed_df = pd.concat([
                    self.transformed_df, subscription_rows_df
                ], ignore_index=True)

        # Add a 'Remark' column for validation
        self.transformed_df['Remark'] = np.nan
        self.transformed_df.loc[(self.transformed_df['Treatment'].isna()) & (self.transformed_df['Expense'] == 0), 'Remark'] = 'Subscription Fee Payment'

        # Drop rows where Treatment is None and Revenue is 0 (duplicates)
        self.transformed_df = self.transformed_df[~((self.transformed_df['Treatment'].isna()) & (self.transformed_df['Revenue'] == 0))]

        self.transformed_df.sort_values(['Period', 'Customer ID'], inplace=True)
        return self.transformed_df

    def aggregate_by_period(self):
        if self.transformed_df is None:
            raise ValueError("The DataFrame has not been transformed yet. Please call transform_to_subscription_scheme first.")

        # Aggregate basic Revenue and Expense
        aggregated_df = self.transformed_df.groupby(self.transformed_df['Period'].dt.to_period("M")).agg({
            'Revenue': 'sum',
            'Expense': 'sum'
        }).reset_index()

        # Add new columns for validation
        patient_status = pd.DataFrame.from_dict(self.patient_pool, orient='index')
        patient_status.reset_index(inplace=True)
        patient_status.rename(columns={'index': 'Customer ID'}, inplace=True)

        aggregated_df['Number of Converted Patients'] = aggregated_df['Period'].apply(
            lambda period: len([cid for cid, details in self.patient_pool.items() 
                                if details['status'] == 'converted' and details['last_seen'] <= period.start_time])
        )
        aggregated_df['Number of Non-Converted Patients'] = aggregated_df['Period'].apply(
            lambda period: len([cid for cid, details in self.patient_pool.items() 
                                if details['status'] in ['non-converted', 'inactive'] and details['last_seen'] <= period.start_time])
        )
        aggregated_df['Number of New Patients'] = aggregated_df['Period'].apply(
            lambda period: len([cid for cid, details in self.patient_pool.items() 
                                if details['last_seen'] == period.start_time])
        )
        aggregated_df['Total Unique Patients Up to Date'] = aggregated_df['Number of Converted Patients'] + aggregated_df['Number of Non-Converted Patients']

        return aggregated_df


