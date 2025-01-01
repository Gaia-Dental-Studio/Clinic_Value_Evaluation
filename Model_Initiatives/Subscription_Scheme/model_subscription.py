import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class ModelSubscriptionScheme:
    def __init__(self, conversion_rate_monthly: float, monthly_subscription_fee: float, 
                 days_until_inactive: int, churn_probability: float):
        """
        Initialize the subscription model transformation class.
        
        Args:
            conversion_rate_monthly (float): Monthly conversion rate (0-1)
            monthly_subscription_fee (float): Subscription fee amount
            days_until_inactive (int): Days until a non-converting customer is considered inactive
            churn_probability (float): Monthly probability of a converting customer churning (0-1)
        """
        self.conversion_rate_monthly = conversion_rate_monthly
        self.monthly_subscription_fee = monthly_subscription_fee
        self.days_until_inactive = days_until_inactive
        self.churn_probability = churn_probability
        
        # Initialize patient pools
        self.converting_patient = set()
        self.new_converting_patient = set()
        self.non_converting_patient = set()
        
        # Track last transaction dates for non-converting patients
        self.last_transaction_dates = {}
        
        # Track subscription dates for converting patients
        self.subscription_dates = {}
        
        # Store original dataframe for revenue restoration
        self.original_df = None
        
        
        self.eligible_treatment = ['015', '018', '013', '114', '121', '141', '022', '521', '941']

    def _is_end_of_month_date(self, date: pd.Timestamp) -> bool:
        """Check if the date is in the last 4 days of the month."""
        next_month = date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        days_until_month_end = (next_month - date).days
        return days_until_month_end <= 4

    def _get_next_subscription_date(self, current_date: pd.Timestamp) -> pd.Timestamp:
        """
        Get the next subscription date, handling end-of-month cases.
        For dates in the last 4 days of the month, use the last day of next month.
        Otherwise, use the same day in the next month.
        """
        if self._is_end_of_month_date(current_date):
            # If it's end of month, use last day of next month
            next_month_start = current_date + pd.offsets.MonthBegin(1)
            return next_month_start + pd.offsets.MonthEnd(1)
        else:
            # Otherwise, try to use the same day next month
            year = current_date.year
            month = current_date.month + 1
            day = current_date.day
            
            if month > 12:
                year += 1
                month = 1
                
            return pd.Timestamp(year=year, month=month, day=day)

    def _generate_subscription_rows(self, customer_id: str, current_month_start: pd.Timestamp) -> list:
        """
        Generate subscription payment row for a customer for the current month only.
        
        Args:
            customer_id (str): Customer ID
            current_month_start (pd.Timestamp): Start of the current month
            
        Returns:
            list: List containing single subscription row DataFrame
        """
        subscription_row = pd.Series({
            'Period': current_month_start,
            'Treatment': 'subscription',
            'Revenue': self.monthly_subscription_fee,
            'Expense': 0,
            'Customer ID': customer_id
        })
        return [pd.DataFrame([subscription_row])]

    def _process_churns(self, current_month_start: pd.Timestamp, month_df: pd.DataFrame) -> tuple[set, pd.DataFrame]:
        """
        Process potential churns for the current month.
        
        Returns:
            tuple: (churned_customers, updated_month_df)
        """
        # Determine which converting customers churn this month
        potential_churners = self.converting_patient.copy()
        num_churners = np.random.binomial(n=len(potential_churners), p=self.churn_probability)
        
        churned_customers = set()
        if num_churners > 0 and potential_churners:
            churned_customers = set(np.random.choice(
                list(potential_churners),
                size=num_churners,
                replace=False
            ))
            
            # Remove churned customers from converting pool
            self.converting_patient.difference_update(churned_customers)
            
            # For churned customers with transactions this month:
            # 1. Restore their original revenue
            # 2. Remove their subscription payment
            if not month_df.empty:
                month_transformed = []
                current_month_str = current_month_start.strftime('%Y-%m')
                
                for _, group in month_df.groupby('Customer ID'):
                    
                    customer_id = group['Customer ID'].iloc[0]
                    
                    if customer_id in churned_customers:
                        # Get original revenue for treatment transactions
                        customer_orig = self.original_df[
                            (self.original_df['Customer ID'] == customer_id) & 
                            (pd.to_datetime(self.original_df['Period']).dt.strftime('%Y-%m') == current_month_str)
                        ]
                        
                        if not customer_orig.empty:
                            # Keep only treatment transactions with original revenue
                            treatment_rows = group[group['Treatment'] != 'subscription'].copy()
                            treatment_rows['Revenue'] = customer_orig['Revenue'].values
                            month_transformed.append(treatment_rows)
                            
                            # print(f"Restoring Revenue for Customer {customer_id} in {current_month_str}")
                            
                    else:
                        month_transformed.append(group)
                
                if month_transformed:
                    month_df = pd.concat(month_transformed, ignore_index=True)
                else:
                    month_df = pd.DataFrame(columns=month_df.columns)
        
        return churned_customers, month_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the transaction dataframe to implement subscription model with churn.
        """
        # Store original dataframe for revenue restoration
        self.original_df = df.copy()
        
        # Convert Period to datetime and sort
        df['Period'] = pd.to_datetime(df['Period'])
        self.original_df['Period'] = pd.to_datetime(self.original_df['Period'])
        df = df.sort_values('Period')
        
        # Get overall date range
        start_date = df['Period'].min()
        end_date = df['Period'].max()
        
        # Initialize transformed dataframe list
        transformed_dfs = []
        
        # Process first month
        first_month = df['Period'].dt.strftime('%Y-%m').iloc[0]
        first_month_df = df[df['Period'].dt.strftime('%Y-%m') == first_month]
        
        first_month_patients = set(first_month_df['Customer ID'].unique())
        
        # Get unique patients for first month
        first_month_patients_eligible = set(first_month_df[first_month_df['Treatment'].isin(self.eligible_treatment)]['Customer ID'].unique())
        
        
        
        # Add all patients to non_converting initially
        self.non_converting_patient.update(first_month_patients)
        
        # Select initial converts
        num_initial_converts = int(len(first_month_patients_eligible) * self.conversion_rate_monthly)
        if num_initial_converts > 0:
            initial_converts = set(np.random.choice(
                list(first_month_patients_eligible),
                size=num_initial_converts,
                replace=False
            ))
            
            # Move to new_converting_patient pool
            self.new_converting_patient.update(initial_converts)
            self.non_converting_patient.difference_update(initial_converts)
        
        # Track active subscribers for each month
        active_subscribers = set()
        
        # Process all months
        for month_year, month_df in df.groupby(df['Period'].dt.strftime('%Y-%m')):
            current_month_start = pd.to_datetime(month_year + '-01')
            current_month_end = current_month_start + pd.offsets.MonthEnd(1)
            
            # Get unique patients with transactions this month
            month_patients = set(month_df['Customer ID'].unique())
            
            month_patients_eligible = set(month_df[month_df['Treatment'].isin(self.eligible_treatment)]['Customer ID'].unique())
            
            # Update active subscribers from previous month
            active_subscribers = (active_subscribers | self.converting_patient | self.new_converting_patient)
            
            # Process churns for this month
            churned_customers, month_df = self._process_churns(current_month_start, month_df)
            
            # Remove churned customers from active subscribers
            active_subscribers = active_subscribers - churned_customers
            
            month_transformed = []
            
            # Reset non_converting pool to only include customers with transactions this month
            # who aren't active subscribers
            self.non_converting_patient = month_patients - active_subscribers
            
            # Add churned customers with transactions back to non_converting pool
            churned_with_transactions = churned_customers.intersection(month_patients)
            if churned_with_transactions:
                self.non_converting_patient.update(churned_with_transactions)
            
            # Select new converts only from customers with transactions this month
            # potential_converts = list(self.non_converting_patient)
            potential_converts = list(month_patients_eligible)
            num_converts = int(len(potential_converts) * self.conversion_rate_monthly)
            
            if potential_converts and num_converts > 0:
                new_converts = set(np.random.choice(
                    potential_converts,
                    size=min(num_converts, len(potential_converts)),
                    replace=False
                ))
                self.new_converting_patient.update(new_converts)
                self.non_converting_patient.difference_update(new_converts)
            
            # Process each customer's transactions
            for customer_id in month_patients:
                customer_df = month_df[month_df['Customer ID'] == customer_id].copy()
                
                if customer_id in active_subscribers:
                    # Existing subscriber
                    # Set treatment revenues to 0
                    
                    for _, row in customer_df.iterrows():
                        if row['Treatment'] in self.eligible_treatment:
                            customer_df.loc[customer_df['Treatment'] == row['Treatment'], 'Revenue'] = 0
                            

                    month_transformed.append(customer_df)
                        
                    # Generate subscription payment for current month
                    subscription_rows = self._generate_subscription_rows(
                        customer_id,
                        current_month_start
                    )
                    month_transformed.extend(subscription_rows)
                    
                elif customer_id in self.new_converting_patient:
                    # New subscriber
                    first_transaction_date = customer_df[customer_df['Customer ID'] == customer_id]['Period'].min()
                    
                    # Store subscription date
                    self.subscription_dates[customer_id] = first_transaction_date
                    

                    for _, row in customer_df.iterrows():
                        if row['Treatment'] in self.eligible_treatment:
                            customer_df.loc[customer_df['Treatment'] == row['Treatment'], 'Revenue'] = 0
                            

                    month_transformed.append(customer_df)
                    
                    # Generate subscription payment for current month
                    subscription_rows = self._generate_subscription_rows(
                        customer_id,
                        current_month_start
                    )
                    month_transformed.extend(subscription_rows)
                else:
                    # Non-converting patient - keep original transactions
                    month_transformed.append(customer_df)
            
            # Generate subscription rows for active subscribers without transactions this month
            subscribers_without_transactions = active_subscribers - month_patients
            for customer_id in subscribers_without_transactions:
                subscription_rows = self._generate_subscription_rows(
                    customer_id,
                    current_month_start
                )
                month_transformed.extend(subscription_rows)
            
            # Move new converts to converting pool for next month
            self.converting_patient.update(self.new_converting_patient)
            self.new_converting_patient.clear()
            
            # Combine all transformations for the month
            if month_transformed:
                month_transformed_df = pd.concat(month_transformed, ignore_index=True)
                transformed_dfs.append(month_transformed_df)
        
        # Combine all months
        result_df = pd.concat(transformed_dfs, ignore_index=True)
        
        # take out rows with subscription treatment
        subscription_df = result_df[result_df['Treatment'] == 'subscription']
        
        # remove subscription rows
        result_df = result_df[result_df['Treatment'] != 'subscription']
        
        return result_df.sort_values(['Period', 'Customer ID']), subscription_df.sort_values(['Period', 'Customer ID'])
    
    
    def get_metrics(self) -> dict:
        """Get current metrics about the transformation."""
        return {
            'total_subscribers': len(self.converting_patient) + len(self.new_converting_patient),
            'active_non_subscribers': len(self.non_converting_patient),
            'new_converts_this_month': len(self.new_converting_patient),
            'established_subscribers': len(self.converting_patient)
        }