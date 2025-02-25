import pandas as pd
import numpy as np

class ModelContributionAmounts:
    def __init__(self, minimum_eligible_amount, apy):
        """
        Initialize the contribution amounts model.

        Parameters:
        - minimum_eligible_amount (float): Minimum treatment cost to be eligible for the program.
        - apy (float): Annual Percentage Yield for interest calculations.
        """
        self.minimum_eligible_amount = minimum_eligible_amount
        self.apy = apy

    def process_transactions(self, transaction_df, conversion_rate, monthly_payment_rate):
        """
        Process transactions to determine contribution program eligibility and generate schedules.

        Parameters:
        - transaction_df (pd.DataFrame): Input transaction DataFrame with 'Customer ID', 'Period', and 'Revenue'.
        - conversion_rate (float): Probability of a transaction being keen for conversion.
        - monthly_payment_rate (float): Percentage of the treatment price as the monthly contribution.

        Returns:
        - pd.DataFrame: Updated transaction DataFrame with a 'Convert' column.
        - pd.DataFrame: Resulting contribution schedule DataFrame.
        """
        
        transaction_df['Period'] = pd.to_datetime(transaction_df['Period'])
        
        # Initialize columns
        transaction_df['Convert'] = False

        # Determine the earliest possible date
        min_period = transaction_df['Period'].min()

        # Initialize result DataFrame
        result_data = []

        for idx, row in transaction_df.iterrows():
            customer_id = row['Customer ID']
            treatment_date = row['Period']
            treatment_price = row['Revenue']

            # Skip if treatment price is below minimum eligible amount
            if treatment_price < self.minimum_eligible_amount:
                continue

            # Apply conversion rate
            if np.random.rand() > conversion_rate:
                continue

            # Calculate monthly payment
            monthly_payment = treatment_price * monthly_payment_rate
            monthly_apy = (1 + self.apy) ** (1 / 12) - 1
            nper = 0
            accrued_interest = 0
            total_future_value = 0
            interest_list = []
            
            # random select integer from 1 to 9 for nper_max
            nper_max = np.random.randint(3, 10) 

            # Calculate nper iteratively to include accrued interest
            while total_future_value < treatment_price and nper <= nper_max:
                

                           
                
                nper += 1
                
                total_future_value += monthly_payment
                interest = total_future_value * monthly_apy
                interest_list.append(interest)
                
                
                # future_value = monthly_payment * (1 + self.apy / 12) ** (nper - 1)
                
                accrued_interest += interest
                
                
                if treatment_price - (total_future_value + accrued_interest) < 0:
                    
                    nper -= 1
                    total_future_value -= monthly_payment
                    accrued_interest -= interest_list[-1]
                    interest_list.pop()
                
                
                    break
                

            
            
            

            

            # Check if nper satisfies the time delta condition
            time_delta = (treatment_date - min_period).days
            if time_delta < 30 * nper:
                continue
            
            # print(f"customer_id: {customer_id}")
            # print(f"total_future_value: {total_future_value}")
            # print(f"treatment_price: {treatment_price}")
            # print(f"accrued_interest: {accrued_interest}")
            # print(f"Monthly Payment: {monthly_payment}")
            
            # print(f"total deposits: {(monthly_payment * nper) + accrued_interest}")
            
            # print(f"nper: {nper}")

            # Eligible transaction, mark as Convert
            transaction_df.at[idx, 'Convert'] = True

            # Generate contribution schedule
            first_deposit_date = treatment_date - pd.Timedelta(days=30 * (nper))
            for i in range(nper):
                deposit_date = first_deposit_date + pd.Timedelta(days=30 * i)
                result_data.append({
                    'Period': deposit_date,
                    'Customer ID': customer_id,
                    'Revenue': monthly_payment,
                    'Expense': 0,
                    'Type': 'Deposit'
                })
                
            for i in interest_list:
                deposit_date = first_deposit_date + pd.Timedelta(days=30 * i)
                result_data.append({
                    'Period': deposit_date,
                    'Customer ID': customer_id,
                    'Revenue': i,
                    'Expense': 0,
                    'Type': 'Interest'
                 })
            

            # Add final payment row
            remaining_fee = treatment_price - (total_future_value + accrued_interest)
            result_data.append({
                'Period': treatment_date,
                'Customer ID': customer_id,
                'Revenue': remaining_fee,
                'Expense': 0,
                'Type': 'Final Payment'
            })
            
            # print(f"remaining_fee: {remaining_fee}")
            # print(f"remaining fee + total deposits: {remaining_fee + (monthly_payment * nper) + accrued_interest}")
            
            # print("---")
            
            
        
        # write me 
        for idx, row in transaction_df.iterrows():
            if row['Convert']:
                result_data.append({
                    'Period': row['Period'],
                    'Customer ID': row['Customer ID'],
                    'Revenue': 0,
                    'Expense': row['Revenue'],
                    'Type': 'Cashback'
                })
        
        
        
        # Create result DataFrame
        result_df = pd.DataFrame(result_data)
        result_df['Period'] = pd.to_datetime(result_df['Period']).dt.date
        
        return transaction_df, result_df

    def validate_result(self, updated_transaction_df, result_df):
        """
        Validate that the result DataFrame correctly calculates deposits and interest.

        Parameters:
        - updated_transaction_df (pd.DataFrame): Transaction DataFrame with 'Convert' column.
        - result_df (pd.DataFrame): Contribution schedule DataFrame.

        Returns:
        - list: Validation issues, if any.
        """
        validation_issues = []

        for customer_id in updated_transaction_df.loc[updated_transaction_df['Convert'] == True, 'Customer ID']:
            # Filter result_df for this customer
            customer_payments = result_df[result_df['Customer ID'] == customer_id]

            # Separate deposit and final payment rows
            deposits = customer_payments[customer_payments['Type'] == 'Deposit']
            final_payment_row = customer_payments[customer_payments['Type'] == 'Final Payment']

            # Recalculate accrued interest
            treatment_date = final_payment_row['Period'].iloc[0]
            total_accrued_interest = 0
            for _, deposit_row in deposits.iterrows():
                deposit_date = deposit_row['Period']
                months_remaining = ((treatment_date - deposit_date).days) // 30
                future_value = deposit_row['Amount'] * (1 + self.apy / 12) ** months_remaining
                accrued_interest = future_value - deposit_row['Amount']
                total_accrued_interest += accrued_interest

            # Calculate totals
            total_deposits = deposits['Amount'].sum()
            final_payment = final_payment_row['Amount'].iloc[0]
            treatment_price = updated_transaction_df.loc[
                (updated_transaction_df['Customer ID'] == customer_id) & (updated_transaction_df['Convert'] == True), 'Revenue'
            ].iloc[0]

            # Compare with treatment price
            calculated_total = total_deposits + total_accrued_interest + final_payment
            if not np.isclose(calculated_total, treatment_price, atol=0.01):
                validation_issues.append({
                    'Customer ID': customer_id,
                    'Total Deposits': total_deposits,
                    'Accrued Interest': total_accrued_interest,
                    'Final Payment': final_payment,
                    'Treatment Price': treatment_price,
                    'Calculated Total': calculated_total
                })

        return validation_issues