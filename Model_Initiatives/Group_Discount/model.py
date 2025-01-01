import pandas as pd
import numpy as np

class ModelGroupDiscount:
    def __init__(self, conversion_rate, max_waiting_days):
        """
        Initialize the model with required parameters.

        Parameters:
        - conversion_rate (float): Probability of a customer opting into the group discount program.
        - max_waiting_days (int): Maximum allowable days between transactions for grouping.
        """
        self.conversion_rate = conversion_rate
        self.max_waiting_days = max_waiting_days

    def process_transactions(self, transactions_df):
        """
        Process the transaction DataFrame to assign Group IDs and adjust dates for grouped transactions.

        Parameters:
        - transactions_df (pd.DataFrame): Input DataFrame with transaction data.

        Returns:
        - pd.DataFrame: DataFrame with updated 'Group ID' and adjusted 'Period' columns.
        """
        transactions_df['Period'] = pd.to_datetime(transactions_df['Period'])
        
        # Ensure data is sorted by Period and Customer ID
        transactions_df = transactions_df.sort_values(by=['Period', 'Customer ID']).reset_index(drop=True)

        # Add a column for conversion probability (is_keen)
        transactions_df['is_keen'] = np.random.rand(len(transactions_df)) < self.conversion_rate

        # Initialize Group ID column
        transactions_df['Group ID'] = np.nan

        # Initialize variables for grouping
        last_group_id = 0

        for idx, row in transactions_df.iterrows():
            if not row['is_keen']:
                # Skip rows where the customer is not keen for the program
                continue

            # Get candidate rows within the max_waiting_days window
            candidates = transactions_df[
                (transactions_df['Period'] <= row['Period']) &
                (transactions_df['Period'] >= row['Period'] - pd.Timedelta(days=self.max_waiting_days)) &
                (transactions_df['Customer ID'] != row['Customer ID']) &
                (transactions_df['Treatment'] == row['Treatment']) &
                (~transactions_df['Group ID'].isna())
            ]

            if not candidates.empty:
                # Assign Group ID of the earliest matching candidate
                group_id = candidates['Group ID'].iloc[0]
            else:
                # Assign a new Group ID if no match exists
                last_group_id += 1
                group_id = last_group_id

            # Update the Group ID for the current row
            transactions_df.at[idx, 'Group ID'] = group_id

        # Adjust dates for grouped transactions
        grouped = transactions_df.dropna(subset=['Group ID']).groupby('Group ID')
        for group_id, group in grouped:
            avg_date = group['Period'].mean()
            transactions_df.loc[transactions_df['Group ID'] == group_id, 'Period'] = avg_date
            
        # make sure the 'Period' column is in datetime format which is not contain hour, minute, second
        transactions_df['Period'] = transactions_df['Period'].dt.date

        # Return the modified DataFrame
        return transactions_df

    def calculate_cashback(self, transactions_df, discount_start, discount_increment, max_group_size=5):
        """
        Calculate cashback (discounts) as expenses for grouped transactions.

        Parameters:
        - transactions_df (pd.DataFrame): Processed transaction DataFrame with 'Group ID' and adjusted 'Period'.
        - discount_start (float): Initial discount rate for groups with at least two members.
        - discount_increment (float): Incremental discount rate for each additional group member beyond two.

        Returns:
        - pd.DataFrame: New DataFrame with 'Period', 'Revenue', and 'Expense' columns for cashback.
        """
        # Filter for rows with valid Group IDs
        grouped = transactions_df.dropna(subset=['Group ID']).groupby('Group ID')

        cashback_data = []

        for group_id, group in grouped:
            group_size = len(group)

            if group_size < 2:
                # Skip groups with fewer than 2 members
                continue
            
            if group_size >= max_group_size:
                # Skip groups with more than 4 members
                continue

            # Calculate the discount for each member in the group
            for _, row in group.iterrows():
                discount_rate = discount_start + (group_size - 2) * discount_increment
                cashback_expense = discount_rate * row['Revenue']
                cashback_data.append({
                    'Period': row['Period'],
                    'Revenue': 0,
                    'Expense': cashback_expense
                })

        # Create a new DataFrame for cashback
        cashback_df = pd.DataFrame(cashback_data)
        
        cashback_df['Period'] = pd.to_datetime(cashback_df['Period']).dt.date
        
        return cashback_df