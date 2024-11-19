import pandas as pd
import streamlit as st 
import random
from cashflow_plot import ModelCashflow
from model_forecasting import ModelForecastPerformance


def loan_amortization_schedule(amount_borrowed, fair_credit_user):
    # Initialize dictionary to hold each customer's schedule
    customer_schedules = {}
    
    # Loop through each customer
    for customer_id in range(1, fair_credit_user + 1):
        # Initialize parameters for each customer
        monthly_principal_payment = amount_borrowed / 10
        monthly_interest_rate = 0.01
        remaining_principal = amount_borrowed

        # Prepare data lists for each customer
        periods = []
        monthly_payments = []
        principal_payments = []
        interest_payments = []
        remaining_principals = []

        # Generate schedule for each period
        for period in range(1, 11):  # 10 periods
            interest_payment = 0 if period == 1 else remaining_principal * monthly_interest_rate
            monthly_payment = monthly_principal_payment + interest_payment
            remaining_principal -= monthly_principal_payment

            # Append data for this period
            periods.append(period)
            monthly_payments.append(monthly_payment)
            principal_payments.append(monthly_principal_payment)
            interest_payments.append(interest_payment)
            remaining_principals.append(remaining_principal)

        # Create DataFrame for this customer and store in the dictionary
        customer_schedule = pd.DataFrame({
            "Period": periods,
            "Monthly Payment": monthly_payments,
            "Principal Payment": principal_payments,
            "Interest Payment": interest_payments,
            "Remaining Principal": remaining_principals
        })
        customer_schedules[customer_id] = customer_schedule

    # Aggregate schedule by summing up all customer schedules
    aggregate_schedule = pd.concat(customer_schedules.values()).groupby("Period").sum().reset_index()

    return customer_schedules, aggregate_schedule



def generate_cashflow_df(customer_schedules, defaulting, cogs_percentage=0.5):
    # Adjust each customer schedule according to the defaulting scenario
    for customer_id, schedule in customer_schedules.items():
        # Check if the customer is defaulting
        if defaulting[customer_id][0] == 0:  # Customer is defaulting
            default_period = defaulting[customer_id][1]
            # Set Monthly Payment to 0 from the period after the default period
            schedule.loc[schedule["Period"] > default_period, "Monthly Payment"] = 0
    
    # Create the aggregate revenue by summing adjusted Monthly Payments across all customers
    aggregate_revenue = pd.concat(customer_schedules.values()).groupby("Period")["Monthly Payment"].sum().reset_index()
    aggregate_revenue.columns = ["Period", "Revenue"]
    
    # Calculate total loan disbursement (Expense) as before, only for the first period
    num_customers = len(customer_schedules)
    total_disbursement = customer_schedules[1]['Principal Payment'].sum() * num_customers
    expense_df = pd.DataFrame({
        "Period": [1],
        "Expense": [total_disbursement * cogs_percentage]
    })

    # Combine Revenue and Expense into the final cashflow DataFrame
    cashflow_df = pd.merge(aggregate_revenue, expense_df, on="Period", how="left").fillna(0)

    return cashflow_df


def generate_defaulting_scenario(num_customers, ratio_defaulting):
    defaulting = {}
    
    for customer_id in range(1, num_customers + 1):
        # Determine if the customer will default (0) or not (1)
        will_default = random.choices([0, 1], weights=[ratio_defaulting, 1 - ratio_defaulting])[0]
        
        if will_default == 0:
            # If they default, assign a random period to start defaulting, from 1 to 9 (ensuring they pay at least once)
            defaulting_period = random.randint(1, 9)
            defaulting[customer_id] = [0, defaulting_period]
        else:
            # If they do not default, mark them as non-defaulting with a single value
            defaulting[customer_id] = [1]
    
    return defaulting

st.title("Fair Credit Financial Services")

st.markdown("## Parameters Assumption")

col1, col2 = st.columns(2)

with col1:
    amount_borrowed = st.number_input("Enter the amount borrowed (Treatment Price)", value=10000)
    fair_credit_user = st.number_input("Enter the number of fair credit users", value=5)

with col2:
    cogs_percentage = st.slider("Select the COGS percentage", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    ratio_defaulting = st.slider("Select the probability of customers defaulting", min_value=0.0, max_value=1.0, value=0.2, step=0.01)


customer_schedules, aggregate_schedule = loan_amortization_schedule(amount_borrowed, fair_credit_user)

defaulting = generate_defaulting_scenario(fair_credit_user, ratio_defaulting)

cashflow = generate_cashflow_df(customer_schedules, defaulting, cogs_percentage)


st.markdown("## Aggregate Payment Schedule")
st.dataframe(aggregate_schedule)

st.divider()

st.markdown("## Expected Cashflow")
st.write("Adjusted for defaulting scenarios, if any")
st.dataframe(cashflow)

cashflow.to_csv('fair_credit_cashflow.csv', index=False)

company_variable = {}

model_cashflow = ModelCashflow()
model = ModelForecastPerformance(company_variable)


cashflow['Period'] = cashflow['Period'].apply(lambda period: model.generate_date_from_month(int(period), method='first_day'))

model_cashflow.add_company_data("Fair Credit", cashflow)
st.plotly_chart(model_cashflow.cashflow_plot(365, granularity='monthly'))

