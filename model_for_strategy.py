import pandas as pd
import numpy as np
import random
import plotly.graph_objs as go

class GeneralModel:
    def create_cashflow_df(self, total_revenue, total_cost, period, period_to_forecast, period_type='monthly', fluctuate=True, denominator=100):
        """
        Create a fluctuative or stable monthly cashflow DataFrame.
        If fluctuate=True, fluctuating values will be generated. 
        If fluctuate=False, values will be evenly distributed across periods.
        
        'period' refers to the time span (in months or years) of the input total_revenue and total_cost.
        """
        # Adjust the total_revenue and total_cost to be monthly values
        if period_type == 'yearly':
            # The input total_revenue and total_cost are for 'period' years, so divide by the number of years
            total_revenue /= (period * 12)  # Convert to monthly revenue
            total_cost /= (period * 12)  # Convert to monthly cost
            forecast_periods = period_to_forecast * 12  # Convert forecast period to months
        elif period_type == 'monthly':
            # The input total_revenue and total_cost are for 'period' months, so divide by the number of months
            total_revenue /= period  # Convert to monthly revenue
            total_cost /= period  # Convert to monthly cost
            forecast_periods = period_to_forecast  # Already in months
        else:
            raise ValueError("Invalid period_type. Must be 'monthly' or 'yearly'.")

        # If fluctuate is True, apply fluctuation logic, otherwise apply equal distribution logic
        if fluctuate:
            # Generate random volumes for the forecasted periods
            random_volumes = np.random.randint(1, denominator, size=forecast_periods-1)  # Random volumes for all but the last period
            final_volume = max(denominator - np.sum(random_volumes), 1)  # Ensure total sums up to denominator and no negatives
            
            # Append the final volume
            volumes = np.append(random_volumes, final_volume)
            
            # Scale the volumes to ensure they sum up to the exact total
            volumes = volumes / np.sum(volumes) * denominator
            
            total_revenue = total_revenue * forecast_periods  # Multiply by the number of forecast periods
            total_cost = total_cost * forecast_periods
            
            # Calculate revenue and expense for each forecast period based on the volumes
            revenue_per_denominator = total_revenue / denominator
            expense_per_denominator = total_cost / denominator
            
            # Use the generated volumes to calculate revenue and expense for each forecasted period
            revenue_list = revenue_per_denominator * volumes 
            expense_list = expense_per_denominator * volumes 
            
        else:
            # If no fluctuation, evenly distribute the total revenue and cost over the forecast periods
            revenue_list = np.full(forecast_periods, total_revenue)  # Equal revenue across months
            expense_list = np.full(forecast_periods, total_cost)  # Equal expense across months

        # Create DataFrame
        cashflow_df = pd.DataFrame({
            'Period': range(1, forecast_periods + 1),  # Reflect months, regardless of whether input was yearly or monthly
            'Revenue': revenue_list,
            'Expense': expense_list
        })
        
        return cashflow_df

class ModelCorporateWellness:
    def __init__(self):
        self.total_potential_employee = 0
        self.conversion_rate = 0
        self.treatments = []
        self.discount_package = 0
        self.subscription_length = 0
        self.total_joining_employee = 0
        
        self.prices_df = None
        self.costs_df = None
        self.dsp_df = None
        
    def set_parameters(self, total_potential_employee, conversion_rate, treatments, discount_package, subscription_length):
        self.total_potential_employee = total_potential_employee
        self.conversion_rate = conversion_rate
        self.treatments = treatments
        self.discount_package = discount_package
        self.subscription_length = subscription_length
        self.total_joining_employee = np.ceil(total_potential_employee * (conversion_rate / 100))
        
    def set_pricing_basis(self, pricing_basis):
        
        if pricing_basis == 'Dr.Riesqi':
                    # Load treatment prices CSV
            self.prices_df = pd.read_csv(r'corporate_wellness_data\treatment_prices.csv')
            
            # Load treatment costs CSV
            self.costs_df = pd.read_csv(r'corporate_wellness_data\treatment_costs.csv')
            self.dsp_df = pd.read_csv(r'corporate_wellness_data\dsp.csv')
        
        elif pricing_basis == 'GAIA Indonesia':
            self.prices_df = pd.read_csv(r'corporate_wellness_data\treatment_prices_GAIA_IDR.csv')
            self.costs_df = pd.read_csv(r'corporate_wellness_data\treatment_costs_GAIA_IDR.csv')
            self.dsp_df = pd.read_csv(r'corporate_wellness_data\dsp_GAIA_IDR.csv')
            
    
    def calculate_ARO(self, treatment_price_df=None, treatment_cost_df=None):
        # Get the price of selected treatments
        selected_treatments = self.treatments

        
        if treatment_price_df is not None:
            # Use the edited treatment prices DataFrame
            self.prices_df = treatment_price_df
        
        # Summing the prices of selected treatments
        total_price = self.prices_df[self.prices_df['Treatment'].isin(selected_treatments)]['Price (Rp.)'].sum()
        
        # Calculate ARO
        aro = (total_price * self.total_joining_employee
               * (100 - self.discount_package) / 100 * (self.subscription_length * 12))
        
        return aro
    
    def calculate_total_cost(self, treatment_cost_df=None):
        
            
        if treatment_cost_df is not None:
            # Use the edited treatment costs DataFrame
            self.costs_df = treatment_cost_df
        
        # Get the cost of selected treatments
        selected_treatments = self.treatments
        
        # Summing the material costs of selected treatments
        material_cost = self.costs_df[self.costs_df['Component'].isin(selected_treatments)]['Material Cost (Rp.)'].sum()
        
        # Calculating total dentist fee based on treatment duration and dentist fee per hour
        self.costs_df['Dentist Fee Total (Rp.)'] = (self.costs_df['Dentist Fee per Hour (Rp.)'] * 
                                                    (self.costs_df['Duration (Min)'] / 60))
        
        dentist_fee_total = self.costs_df[self.costs_df['Component'].isin(selected_treatments)]['Dentist Fee Total (Rp.)'].sum()
        
        # Get the card fee (remains unchanged)
        card_fee = self.costs_df[self.costs_df['Component'] == 'Member Card (monthly)']['Material Cost (Rp.)'].values[0]
        
        # Total cost per employee is now the sum of material cost, dentist fee, and monthly card fee
        total_cost_per_employee = material_cost + dentist_fee_total + card_fee
        
        # Total cost is multiplied by the number of joining employees and subscription length
        total_cost = (total_cost_per_employee * self.total_joining_employee
                    * (self.subscription_length * 12))
        
        return total_cost


    def calculate_DSP(self, dsp_editor_df, total_joining_employee):
        # Get the selected DSP treatments and their conversion rates and discount rates from the edited DSP editor
        dsp_selected = dsp_editor_df[dsp_editor_df['Selected'] == True]
        
        dsp_original_df = dsp_selected  # we will use the selected as it needs to be editable
        
        total_dsp_aro = 0
        total_dsp_cost = 0
        
        dsp_output_data = {
            "Treatment": [],
            "Joining Customers": [],
            "Total Revenue (Rp.)": [],
            "Total Cost (Rp.)": []
        }
        
        # Iterate through the selected rows in dsp_editor_df
        for _, row in dsp_selected.iterrows():
            treatment_name = row['Treatment']
            dsp_conversion_rate = row['Conversion Rate (%)'] / 100
            discount_rate = row['Discount Price (%)'] / 100
            
            # Cross-reference with original dsp_df to get Original Price and Cost Material
            original_row = dsp_original_df[dsp_original_df['Treatment'] == treatment_name].iloc[0]
            original_price = original_row['Original Price (Rp.)']
            dsp_cost_material = original_row['Cost Material (Rp.)']
            duration_min = original_row['Duration (Min)']
            dentist_fee_per_hour = original_row['Dentist Fee Per Hour (Rp.)']
            
            # Calculate the new discounted price
            dsp_price = original_price * (1 - discount_rate)
            
            # Calculate the dentist fee based on duration and fee per hour
            dsp_dentist_fee = (dentist_fee_per_hour * (duration_min / 60))
            
            dsp_total_joining = np.ceil(total_joining_employee * dsp_conversion_rate)
            
            # DSP ARO and cost calculations
            dsp_aro = dsp_price * dsp_total_joining
            dsp_cost = (dsp_cost_material + dsp_dentist_fee) * dsp_total_joining
            
            total_dsp_aro += dsp_aro
            total_dsp_cost += dsp_cost
            
            # Add row to output data
            dsp_output_data["Treatment"].append(treatment_name)
            dsp_output_data["Joining Customers"].append(int(dsp_total_joining))
            dsp_output_data["Total Revenue (Rp.)"].append(int(dsp_aro))
            dsp_output_data["Total Cost (Rp.)"].append(int(dsp_cost))
        
        # Create a DataFrame for output
        dsp_df_output = pd.DataFrame(dsp_output_data)
        
        return total_dsp_aro, total_dsp_cost, dsp_df_output
    

    # def _convert_price(self, price):
    #     """Helper function to convert price from string to integer."""
    #     return int(price.replace('Rp', '').replace('.', '').replace(',', '').strip())
    
    


    def run_sensitivity_analysis(self, variables, increments):
        results = []

        # Extract initial values for the selected variables
        initial_values = {
            "Conversion Rate (%)": self.conversion_rate,
            "Discount Package (%)": self.discount_package,
            "Total Potential Employee": self.total_potential_employee
        }

        # Generate arrays for both variables with 10 points up and down
        value_arrays = {}
        for i, var in enumerate(variables):
            initial_value = initial_values[var]
            increment = increments[i]

            # Create an array for increasing values, bounded by the max limit (100 for percentages)
            up_values = [initial_value + increment * n for n in range(1, 11) if initial_value + increment * n <= 100]

            # Create an array for decreasing values, bounded by the min limit (0 for percentages)
            down_values = [initial_value - increment * n for n in range(1, 11) if initial_value - increment * n >= 0]

            value_arrays[var] = sorted(down_values + [initial_value] + up_values)

        # Iterate over all combinations of variable values
        for value1 in value_arrays[variables[0]]:
            for value2 in value_arrays[variables[1]]:
                # Update the object's attributes based on the selected variables
                if variables[0] == "Conversion Rate (%)":
                    self.conversion_rate = value1
                elif variables[0] == "Discount Package (%)":
                    self.discount_package = value1
                elif variables[0] == "Total Potential Employee":
                    self.total_potential_employee = value1

                if variables[1] == "Conversion Rate (%)":
                    self.conversion_rate = value2
                elif variables[1] == "Discount Package (%)":
                    self.discount_package = value2
                elif variables[1] == "Total Potential Employee":
                    self.total_potential_employee = value2
                    
                self.total_joining_employee = np.ceil(self.total_potential_employee * (self.conversion_rate / 100))

                # Recalculate the financial metrics
                first_aro_revenue = self.calculate_ARO()
                total_dsp_aro, total_dsp_cost, dsp_df_output = self.calculate_DSP(self.dsp_df, self.total_joining_employee)
                total_revenue = first_aro_revenue + total_dsp_aro
                
                first_cost = round(self.calculate_total_cost(), 0)
                total_cost = round(first_cost + total_dsp_cost, 0)

                total_profit = total_revenue - total_cost

                # Store the results in a list
                results.append([value1, value2, total_revenue, total_cost ,total_profit])

        # Create the results DataFrame
        results_df = pd.DataFrame(results, columns=[variables[0], variables[1], 'Total Revenue' , 'Total Cost' ,'Total Profit'])

        # Pivot the DataFrame to create a matrix format suitable for a heatmap
        profit_matrix = results_df.pivot(index=variables[0], columns=variables[1], values='Total Profit')

        # Create the heatmap using Plotly, including the profit values as text in each cell
        fig = go.Figure(data=go.Heatmap(
            z=profit_matrix.values,
            x=profit_matrix.columns,
            y=profit_matrix.index,
            colorscale='Viridis',  # You can choose other color scales like 'Plasma', 'Cividis', etc.
            colorbar=dict(title='Total Profit'),
            text=profit_matrix.values,
            texttemplate="%{text:,}",  # Format the text to include thousand separators and no decimals
            textfont={"size": 10}  # Adjust the size to make it more readable
        ))

        fig.update_layout(
            title='Sensitivity Analysis of Total Profit',
            xaxis_title=variables[1],
            yaxis_title=variables[0],
            template='plotly_white'
        )

        return fig, results_df







class ModelSchoolOutreach:
    def __init__(self, converting_students, converting_parents, converting_students_no_parents, discount_single, discount_family):
        self.converting_students = converting_students
        self.converting_parents = converting_parents
        self.total_converting = converting_students + converting_parents
        
        self.converting_students_no_parents = converting_students_no_parents
        self.converting_students_with_parents = self.total_converting - converting_students_no_parents
        
        # Discount prices for Single and Family groups
        self.discount_single = discount_single
        self.discount_family = discount_family
        
        # Load the treatment prices CSV
        self.treatment_prices_df = pd.read_csv(r'school_outreach_data\treatment_prices_new.csv')
        self.event_cost_df = pd.read_csv(r'school_outreach_data\event_cost.csv')

    def initial_price_df(self):
        # Prepare the DataFrame and add discount columns
        df = self.treatment_prices_df.copy()
        df['Discount Single (%)'] = self.discount_single
        df['Discount Family (%)'] = self.discount_family
        
        # drop Dentist Fee (Rp.) column
        df = df.drop(columns=['Dentist Fee (Rp.)'])
        
        return df
    
    def calculate_package_demand(self, package_list_df):
        # Create a copy of the package list dataframe to avoid modifying the original
        df = package_list_df.copy()

        # Calculate Demand with Single Discount, set to 0 if 'Category' is 'Parent'
        df['Demand with Single Discount'] = df.apply(
            lambda row: self.converting_students_no_parents * (row['Conversion Rate Single (%)'] / 100) if row['Category'] == 'Student' else 0,
            axis=1
        )
        
        df['Demand with Single Discount'] = df['Demand with Single Discount'].astype(int)

        # Calculate Demand with Family Discount for both categories
        df['Demand with Family Discount'] = self.converting_students_with_parents * (df['Conversion Rate Family (%)'] / 100)
        df['Demand with Family Discount'] = df['Demand with Family Discount'].astype(int)

        # Calculate Total Demand as the sum of Single and Family demand
        df['Total Demand'] = df['Demand with Single Discount'] + df['Demand with Family Discount']

        # Select and return the required columns
        result_df = df[['Treatment Package', 'Category', 'Description', 'Demand with Single Discount', 'Demand with Family Discount', 'Total Demand']].copy()

        return result_df


    def price_df(self, df):
        # Adjust prices for both groups (Single and Family)
        df['Adjusted Price Single (Rp.)'] = df['Original Price (Rp.)'] * (1 - (df['Discount Single (%)'] / 100))
        df['Adjusted Price Family (Rp.)'] = df['Original Price (Rp.)'] * (1 - (df['Discount Family (%)'] / 100))
        
        # Calculate the new dentist fee based on total duration and fee per hour
        df['Dentist Fee (Rp.)'] = (df['total_duration'] / 60) * df['Dentist Fee per Hour']
        
        # Select and rename columns to create the final price_df
        price_df = df[['Treatment', 'Category', 'Adjusted Price Single (Rp.)', 'Adjusted Price Family (Rp.)', 
                    'Unit of Measure', 'Cost Material (Rp.)', 'Dentist Fee (Rp.)']].copy()
        
        return price_df

    def calculate_financials(self, price_df, package_demand_df, total_event_cost, event_frequency):
        # Initialize total revenue, total cost, and total profit variables
        total_revenue = 0
        total_cost = 0
        total_profit = 0
        
        # Iterate over each package in package_demand_df
        for _, package_row in package_demand_df.iterrows():
            # Get the package treatments from the Description column
            package_treatments = package_row['Description'].split(', ')
            
            # Get the demands for single and family discount
            demand_single = package_row['Demand with Single Discount']
            demand_family = package_row['Demand with Family Discount']
            
            # Filter the price_df to get the relevant treatments for this package
            for treatment in package_treatments:
                treatment_price_row = price_df[(price_df['Treatment'] == treatment) & (price_df['Category'] == package_row['Category'])]
                
                if not treatment_price_row.empty:
                    # Extract prices and costs
                    adjusted_price_single = treatment_price_row['Adjusted Price Single (Rp.)'].values[0]
                    adjusted_price_family = treatment_price_row['Adjusted Price Family (Rp.)'].values[0]
                    cost_material = treatment_price_row['Cost Material (Rp.)'].values[0]
                    dentist_fee = treatment_price_row['Dentist Fee (Rp.)'].values[0]
                    
                    # Calculate revenue for single and family discounts
                    revenue_single = adjusted_price_single * demand_single
                    revenue_family = adjusted_price_family * demand_family
                    
                    # Calculate cost for single and family discounts
                    cost_single = (cost_material + dentist_fee) * demand_single
                    cost_family = (cost_material + dentist_fee) * demand_family
                    
                    # Calculate profit for single and family discounts
                    profit_single = revenue_single - cost_single
                    profit_family = revenue_family - cost_family
                    
                    # Add to totals
                    total_revenue += revenue_single + revenue_family
                    total_cost += cost_single + cost_family
                    total_profit += profit_single + profit_family
        
        # Add event costs to total cost
        total_cost += total_event_cost * event_frequency
        
        # Return the overall financials
        return total_revenue, total_cost, total_profit



    def initial_event_cost_df(self):
        return self.event_cost_df
    





class ModelAgecareOutreach:
    def __init__(self, total_population, conversion_rate, discount_price):

        self.total_population = total_population
        self.conversion_rate = conversion_rate
        self.discount_price = discount_price
        self.total_joined = self.total_population * (self.conversion_rate / 100)
        
        
        # Load the treatment prices CSV
        self.treatment_prices_df = pd.read_csv(r'agecare_outreach_data\treatment_prices.csv')
        self.event_cost_df = pd.read_csv(r'agecare_outreach_data\event_cost.csv')

    
    def initial_price_df(self):
        df = self.treatment_prices_df.copy()
        df['Discount Price (%)'] = self.discount_price
        return df
        
    
    
    def price_df(self, df):
        # Adjust prices based on discount price
        df['Adjusted Price (Rp.)'] = df['Original Price (Rp.)'] * ( 1 -  (df['Discount Price (%)'] / 100))
        
        
        
        # Create the DataFrame with necessary columns
        price_df = df.copy()
        price_df['Adjusted Price (Rp.)'] = df['Adjusted Price (Rp.)']
        price_df['Demand'] = np.ceil(self.total_joined * (df['Conversion Rate (%)'] / 100))
        # price_df = price_df[['Treatment', 'Adjusted Price (Rp.)', 'Demand']]
        
        return price_df
    
    def calculate_financials(self, price_df, total_event_cost, event_frequency):
        # Calculate total revenue for each treatment
        price_df['Total Revenue (Rp.)'] = price_df['Adjusted Price (Rp.)'] * price_df['Demand']
        
        # Calculate total cost for each treatment (Cost Material + Dentist Fee) * Demand
        price_df['Total Cost (Rp.)'] = (price_df['Cost Material (Rp.)'] + price_df['Dentist Fee (Rp.)']) * price_df['Demand']
        
        # Calculate total profit for each treatment (Total Revenue - Total Cost)
        price_df['Total Profit (Rp.)'] = price_df['Total Revenue (Rp.)'] - price_df['Total Cost (Rp.)']
        
        # Sum total revenue, total cost, and total profit across all treatments
        total_revenue = price_df['Total Revenue (Rp.)'].sum()
        total_cost = price_df['Total Cost (Rp.)'].sum() + total_event_cost * event_frequency
        total_profit = price_df['Total Profit (Rp.)'].sum()

        # Return the overall financials and the price_df with detailed calculations
        return total_revenue, total_cost, total_profit
    
    def initial_event_cost_df(self):
        
        return self.event_cost_df
    
    
class ModelSpecialNeedsOutreach:
    def __init__(self, total_population, conversion_rate, discount_price):

        self.total_population = total_population
        self.conversion_rate = conversion_rate
        self.discount_price = discount_price
        self.total_joined = self.total_population * (self.conversion_rate / 100)
        
        
        # Load the treatment prices CSV
        self.treatment_prices_df = pd.read_csv(r'special_needs_outreach_data\treatment_prices.csv')
        self.event_cost_df = pd.read_csv(r'special_needs_outreach_data\event_cost.csv')

    
    def initial_price_df(self):
        df = self.treatment_prices_df.copy()
        df['Discount Price (%)'] = self.discount_price
        return df
        
    
    
    def price_df(self, df):
        # Adjust prices based on discount price
        df['Adjusted Price (Rp.)'] = df['Original Price (Rp.)'] * ( 1 -  (df['Discount Price (%)'] / 100))
        
        
        
        # Create the DataFrame with necessary columns
        price_df = df.copy()
        price_df['Adjusted Price (Rp.)'] = df['Adjusted Price (Rp.)']
        price_df['Demand'] = np.ceil(self.total_joined * (df['Conversion Rate (%)'] / 100))
        # price_df = price_df[['Treatment', 'Adjusted Price (Rp.)', 'Demand']]
        
        return price_df
    
    def calculate_financials(self, price_df, total_event_cost, event_frequency):
        # Calculate total revenue for each treatment
        price_df['Total Revenue (Rp.)'] = price_df['Adjusted Price (Rp.)'] * price_df['Demand']
        
        # Calculate total cost for each treatment (Cost Material + Dentist Fee) * Demand
        price_df['Total Cost (Rp.)'] = (price_df['Cost Material (Rp.)'] + price_df['Dentist Fee (Rp.)'] + price_df['Sedation Cost (Rp.)']) * price_df['Demand']
        
        # Calculate total profit for each treatment (Total Revenue - Total Cost)
        price_df['Total Profit (Rp.)'] = price_df['Total Revenue (Rp.)'] - price_df['Total Cost (Rp.)']
        
        # Sum total revenue, total cost, and total profit across all treatments
        total_revenue = price_df['Total Revenue (Rp.)'].sum()
        total_cost = price_df['Total Cost (Rp.)'].sum() + total_event_cost * event_frequency
        total_profit = price_df['Total Profit (Rp.)'].sum()

        # Return the overall financials and the price_df with detailed calculations
        return total_revenue, total_cost, total_profit
    
    def initial_event_cost_df(self):
        
        return self.event_cost_df