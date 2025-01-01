import pandas as pd
import numpy as np

class ModelFamilyDiscount:
    def __init__(self, minimum_eligible_spending):
        """
        Initialize the family discount model.

        Parameters:
        - minimum_eligible_spending (float): Minimum spending required for a customer to become eligible for discounts.
        """
        self.minimum_eligible_spending = minimum_eligible_spending

    def create_dummy_family_df(self, transaction_df, pmf_family_sizes):
        """
        Create a dummy Family DataFrame with randomized Family IDs and initial eligibility set to False.

        Parameters:
        - transaction_df (pd.DataFrame): Input transaction DataFrame with 'Customer ID'.
        - pmf_family_sizes (dict): Probability mass function for family sizes.

        Returns:
        - pd.DataFrame: Family DataFrame with 'Customer ID', 'Family ID', and 'eligible' columns.
        """
        transaction_df['Period'] = pd.to_datetime(transaction_df['Period'])  # Ensure 'Period' is in datetime format
        
        # Extract unique Customer IDs
        unique_customers = transaction_df['Customer ID'].unique()
        np.random.shuffle(unique_customers)

        # Initialize the Family DataFrame
        family_df = pd.DataFrame({'Customer ID': unique_customers, 'Family ID': np.nan, 'eligible': False})

        # Generate family sizes based on the provided PMF
        family_sizes = np.random.choice(
            list(pmf_family_sizes.keys()),
            size=len(unique_customers),
            p=list(pmf_family_sizes.values())
        )

        # Assign Family IDs
        current_index = 0
        family_id = 1
        while current_index < len(unique_customers):
            family_size = family_sizes[current_index]
            end_index = min(current_index + family_size, len(unique_customers))
            family_df.loc[current_index:end_index - 1, 'Family ID'] = family_id
            family_id += 1
            current_index = end_index

        return family_df

    def process_transactions(self, transaction_df, family_df, discount_start, discount_increment, max_discount_rate=0.3):
        """
        Process transactions row by row to update eligibility and calculate cashback.

        Parameters:
        - transaction_df (pd.DataFrame): Input transaction DataFrame with 'Customer ID', 'Period', and 'Revenue'.
        - family_df (pd.DataFrame): Family DataFrame with initial 'Family ID' and 'eligible' columns.
        - discount_start (float): Initial discount rate.
        - discount_increment (float): Incremental discount rate for each additional eligible member.

        Returns:
        - pd.DataFrame: Cashback DataFrame with 'Period', 'Revenue', and 'Expense'.
        """
        # Sort transaction_df by Period
        transaction_df = transaction_df.sort_values(by='Period').reset_index(drop=True)

        # Initialize aggregated spending tracker
        aggregated_spending = {customer_id: 0 for customer_id in family_df['Customer ID']}

        # Initialize cashback data list
        cashback_data = []

        for idx, row in transaction_df.iterrows():
            customer_id = row['Customer ID']
            period = row['Period']
            revenue = row['Revenue']

            # Update aggregated spending for this customer within the 1-year window
            valid_transactions = transaction_df[
                (transaction_df['Customer ID'] == customer_id) &
                (transaction_df['Period'] >= period - pd.Timedelta(days=365)) &
                (transaction_df['Period'] <= period)
            ]
            aggregated_spending[customer_id] = valid_transactions['Revenue'].sum()

            # Check if customer is eligible and update family_df
            if aggregated_spending[customer_id] >= self.minimum_eligible_spending:
                family_df.loc[family_df['Customer ID'] == customer_id, 'eligible'] = True

            # Count eligible family members
            family_id = family_df.loc[family_df['Customer ID'] == customer_id, 'Family ID'].values[0]
            family_group = family_df[family_df['Family ID'] == family_id]
            eligible_members = family_group[family_group['eligible']]
            eligible_group_size = len(eligible_members)

            # Calculate cashback if at least 2 family members are eligible
            if eligible_group_size >= 2:
                discount_rate = discount_start + (eligible_group_size - 2) * discount_increment 
                discount_rate = min(discount_rate, max_discount_rate)
                cashback_expense = discount_rate * revenue
                cashback_data.append({
                    'Period': period,
                    'Revenue': 0,
                    'Expense': cashback_expense
                })

        # Create a cashback DataFrame
        cashback_df = pd.DataFrame(cashback_data)
        return cashback_df