import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px


class ModelClinicValue:
    def __init__(self, company_variables):
        # Extract scalar values from the company variables dictionary
        self.net_sales = company_variables.get('Net Sales', 0)
        self.cogs = company_variables.get('COGS', 0)
        self.trading_income = company_variables.get('Trading Income', 0)
        self.other_income = company_variables.get('Other Income', 0)
        self.interest_revenue = company_variables.get('Interest Revenue of Bank', 0)
        self.operational_expense = company_variables.get('Operational Expense', 0)
        self.other_expense = company_variables.get('Other Expense', 0)
        self.clinic_depreciation = company_variables.get('Depreciation of Equipment Clinic Expense', 0)
        self.non_clinic_depreciation = company_variables.get('Depreciation of Equipment Non Clinic Expense', 0)
        self.bank_tax_expense = company_variables.get('Bank Tax Expense', 0)
        self.other_tax = company_variables.get('Other Tax', 0)
        self.equipment_value = company_variables.get('Equipments Value', 0)
        self.general_expense = self.operational_expense + self.other_expense
        self.ebitda = self.net_sales + self.cogs + self.trading_income + self.other_income + self.general_expense
        self.depreciation_total = self.clinic_depreciation + self.non_clinic_depreciation
        self.tax_total = self.bank_tax_expense + self.other_tax
        self.revenue = self.net_sales + self.trading_income + self.other_income
        self.ebit = self.ebitda + self.depreciation_total #+ self.tax_total
        self.ebit_ratio = self.ebit / self.revenue if self.revenue > 0 else None
        self.net_sales_growth = company_variables.get('Net Sales Growth', 0)
        self.relative_variability_net_sales = company_variables.get('Relative Variation of Net Sales', 0)
        self.number_of_patients = company_variables.get('Number of Active Patients', 0)
        self.relative_variability_patient_spending = company_variables.get('Relative Variation of Patient Spending', 0)
        self.potential_existing_dentist_leaving = company_variables.get('Potential Existing Dentist Leaving', 0)
        # self.number_of_dentist = company_variables.get('Current Number of Dentist', 0)
        # self.projected_number_of_dentist = company_variables.get('Projected Number of Dentist', 0)
        
        # convert to integer for self.number_of_dentist and self.projected_number_of_dentist
        # self.number_of_dentist = int(self.number_of_dentist)
        # self.projected_number_of_dentist = int(self.projected_number_of_dentist)
        self.number_of_patients = int(self.number_of_patients)
        

        # Extract DataFrames from the company variables dictionary
        self.net_cash_flow = company_variables.get('Net Cash Flow', None)
        self.equipment_life = company_variables.get('Equipment_Life', pd.DataFrame())
        
        self.dentist_contribution = None
        self.net_sales_yearly = None
        self.net_sales_monthly = None
        self.patient_transaction = None

    # def analyze_cash_flow(self):
    #     # Replace None values with 0 in the net_cash_flow DataFrame
    #     self.net_cash_flow = self.net_cash_flow.fillna(0)

    #     # Transform the wide-format net_cash_flow DataFrame into a long format
    #     long_data = self.net_cash_flow.reset_index().melt(id_vars='index', var_name='Month', value_name='Net Cashflow')
    #     long_data.rename(columns={'index': 'Year'}, inplace=True)
    #     long_data['Year'] = long_data['Year'].astype(int)

    #     # Sort data by Year and Month
    #     month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    #     long_data['Month'] = pd.Categorical(long_data['Month'], categories=month_order, ordered=True)
    #     long_data = long_data.sort_values(['Year', 'Month']).reset_index(drop=True)

    #     # Convert Net Cashflow values to numeric, forcing errors to NaN
    #     long_data['Net Cashflow'] = pd.to_numeric(long_data['Net Cashflow'], errors='coerce')

    #     # Drop rows with missing values (None or NaN)
    #     cleaned_data = long_data.dropna(subset=['Net Cashflow'])

    #     # Calculate the standard deviation of the Net Cashflow
    #     std_deviation = cleaned_data['Net Cashflow'].std()
        
    #     # Calculate the average of the Net Cashflow
    #     average_cashflow = cleaned_data['Net Cashflow'].mean()

    #     # Calculate the trend line coefficient (slope) using linear regression
    #     if not cleaned_data.empty:
    #         cleaned_data['Time_Index'] = np.arange(len(cleaned_data))  # Create a time index for the regression
    #         X = cleaned_data[['Time_Index']]
    #         y = cleaned_data['Net Cashflow']
    #         model = LinearRegression()
    #         model.fit(X, y)
    #         trend_coefficient = model.coef_[0]
    #     else:
    #         trend_coefficient = 0  # If there's no data, return 0 for the trend coefficient

    #     return average_cashflow, std_deviation, trend_coefficient
    
    def calculate_equipment_usage_ratio(self):
        # Filter the equipment that the clinic owns (where 'Own?' is True)
        owned_equipment = self.equipment_life[self.equipment_life['Own?'] == True].copy()
        
        owned_equipment['Current Lifetime Usage'] = owned_equipment.apply(lambda row: row['Current Lifetime Usage'] if row['Current Lifetime Usage'] <= row['Expected Lifetime'] else row['Expected Lifetime'], axis=1)

        # Create new columns for total expected lifetime and total current lifetime usage
        owned_equipment['Total Expected Lifetime'] = owned_equipment['Quantity'] * owned_equipment['Expected Lifetime']
        owned_equipment['Total Current Lifetime Usage'] = owned_equipment['Quantity'] * owned_equipment['Current Lifetime Usage']
        
        owned_equipment['Depreciation per Year'] = owned_equipment['Price'] / owned_equipment['Expected Lifetime']
        owned_equipment['Total Depreciation'] = owned_equipment['Depreciation per Year'] * owned_equipment['Current Lifetime Usage']
        owned_equipment['Remaining Value'] = owned_equipment['Price'] - owned_equipment['Total Depreciation']

        # Calculate the sum of Total Expected Lifetime and Total Current Lifetime Usage
        total_expected_lifetime_sum = owned_equipment['Total Expected Lifetime'].sum()
        total_current_lifetime_usage_sum = owned_equipment['Total Current Lifetime Usage'].sum()
        total_remaining_value = owned_equipment['Remaining Value'].sum()
        

        # Calculate the Equipment Usage Ratio
        equipment_usage_ratio =  total_current_lifetime_usage_sum / total_expected_lifetime_sum if total_current_lifetime_usage_sum > 0 else None
        total_equipments = owned_equipment['Quantity'].sum()


        return equipment_usage_ratio, total_equipments, total_remaining_value
    
    def equipment_adjusting_value(self, total_remaining_value, baseline_tangible_assets, baseline_usage_ratio):
        
        baseline_depreciation_total = baseline_tangible_assets * baseline_usage_ratio

        
        adjustment_value = total_remaining_value - (baseline_tangible_assets - baseline_depreciation_total)
        
        return adjustment_value
    

    def update_variable_from_uploaded_file(self, uploaded_file):
        # Read the Excel file and extract data from the 'main_variables' sheet
        main_variables_df = pd.read_excel(uploaded_file, sheet_name='main_variables')

        # Check if the DataFrame has the necessary columns
        if 'Variable' in main_variables_df.columns and 'Value' in main_variables_df.columns:
            # Create a dictionary from the DataFrame for quick lookup
            variable_dict = dict(zip(main_variables_df['Variable'], main_variables_df['Value']))

            # Update each variable with its corresponding value from the file, or default to 0 if not found
            self.net_sales = variable_dict.get('Net Sales', 0)
            self.cogs = variable_dict.get('COGS', 0)
            self.trading_income = variable_dict.get('Trading Income', 0)
            self.other_income = variable_dict.get('Other Income', 0)
            self.interest_revenue = variable_dict.get('Interest Revenue of Bank', 0)
            self.operational_expense = variable_dict.get('Operational Expense', 0)
            self.other_expense = variable_dict.get('Other Expense', 0)
            self.clinic_depreciation = variable_dict.get('Depreciation of Equipment Clinic Expense', 0)
            self.non_clinic_depreciation = variable_dict.get('Depreciation of Equipment Non Clinic Expense', 0)
            self.bank_tax_expense = variable_dict.get('Bank Tax Expense', 0)
            self.other_tax = variable_dict.get('Other Tax', 0)
            self.equipment_value = variable_dict.get('Equipments Value', 0)
            self.general_expense = self.operational_expense + self.other_expense
            self.ebitda = self.net_sales + self.cogs + self.trading_income + self.other_income + self.general_expense
            self.depreciation_total = self.clinic_depreciation + self.non_clinic_depreciation
            self.tax_total = self.bank_tax_expense + self.other_tax
            self.ebit = variable_dict.get('EBIT', self.ebitda + self.depreciation_total) #+ self.tax_total)
            self.revenue = self.net_sales + self.trading_income + self.other_income
            self.ebit_ratio = variable_dict.get('EBIT Ratio', self.ebit / self.revenue if self.revenue > 0 else None)
            self.net_sales_growth = variable_dict.get('Net Sales Growth', 0)
            self.relative_variability_net_sales = variable_dict.get('Relative Variation of Net Sales', 0)
            self.number_of_patients = variable_dict.get('Number of Active Patients', 0)
            self.relative_variability_patient_spending = variable_dict.get('Relative Variation of Patient Spending', 0)
            self.potential_existing_dentist_leaving = variable_dict.get('Potential Existing Dentist Leaving', 0)
            # self.number_of_dentist = variable_dict.get('Current Number of Dentist', 0)
            # self.projected_number_of_dentist = variable_dict.get('Projected Number of Dentist', 0)
            
            # self.number_of_dentist = int(self.number_of_dentist)
            # self.projected_number_of_dentist = int(self.projected_number_of_dentist)
            self.number_of_patients = int(self.number_of_patients)
            
        self.dentist_contribution = pd.read_excel(uploaded_file, sheet_name='dentist_contribution')
        self.net_sales_yearly = pd.read_excel(uploaded_file, sheet_name='yearly_net_sales')
        self.net_sales_monthly = pd.read_excel(uploaded_file, sheet_name='monthly_net_sales')
        self.patient_transaction = pd.read_excel(uploaded_file, sheet_name='patient_transaction')

        # # Read the net_cash_flow data from the 'net_cash_flow' sheet
        # net_cash_flow_df = pd.read_excel(uploaded_file, sheet_name='net_cash_flow', index_col=0)

        # # Ensure that the index (year) is properly formatted as integers
        # net_cash_flow_df.index = net_cash_flow_df.index.astype(int)

        # # Ensure that the DataFrame contains all the default years ['2019', '2020', '2021', '2022', '2023']
        # default_years = [2019, 2020, 2021, 2022, 2023]
        # for year in default_years:
        #     if year not in net_cash_flow_df.index:
        #         # Add missing years with None values for all months
        #         net_cash_flow_df.loc[year] = [None] * 12

        # # Reorder the index to maintain the order of default years
        # net_cash_flow_df = net_cash_flow_df.reindex(default_years)

        # self.net_cash_flow = net_cash_flow_df
        
    def ebit_baseline_to_multiple(self, net_sales_growth):
        
        ebit = self.ebit
        ebit_ratio = self.ebit_ratio
        
        data = pd.read_csv('EBIT_baseline_to_multiple.csv')
        # Define boundary values based on the unique values in the dataset
        ebit_boundaries = sorted(data['EBIT'].unique())
        ebit_ratio_boundaries = sorted(data['EBIT Ratio'].unique())
        net_sales_growth_boundaries = sorted(data['Net Sales Growth'].unique())

        # Round down to nearest boundary value, with a fallback to the minimum boundary
        ebit = max([val for val in ebit_boundaries if val <= ebit], default=min(ebit_boundaries))
        ebit_ratio = max([val for val in ebit_ratio_boundaries if val <= ebit_ratio], default=min(ebit_ratio_boundaries))
        net_sales_growth = max([val for val in net_sales_growth_boundaries if val <= net_sales_growth], default=min(net_sales_growth_boundaries))

        # Lookup the row with the closest match
        result = data[(data['EBIT'] == ebit) & 
                    (data['EBIT Ratio'] == ebit_ratio) & 
                    (data['Net Sales Growth'] == net_sales_growth)]

        # Return the EBIT Multiple if a match is found, else return None
        if not result.empty:
            return result['EBIT Multiple'].values[0]
        else:
            return None



    # def ebit_baseline_to_multiple(self, net_sales_growth):
        
    #     ebit = self.ebit
    #     ebit_ratio = self.ebit_ratio
        
        
    #     data = pd.read_csv('EBIT_baseline_to_multiple.csv')
    #     # Define boundary values based on the unique values in the dataset
    #     ebit_boundaries = sorted(data['EBIT'].unique())
    #     ebit_ratio_boundaries = sorted(data['EBIT Ratio'].unique())
    #     net_sales_growth_boundaries = sorted(data['Net Sales Growth'].unique())

    #     # Round down to nearest boundary value
    #     ebit = max([val for val in ebit_boundaries if val <= ebit])
    #     ebit_ratio = max([val for val in ebit_ratio_boundaries if val <= ebit_ratio])
    #     net_sales_growth = max([val for val in net_sales_growth_boundaries if val <= net_sales_growth])

    #     # Lookup the row with the closest match
    #     result = data[(data['EBIT'] == ebit) & 
    #                 (data['EBIT Ratio'] == ebit_ratio) & 
    #                 (data['Net Sales Growth'] == net_sales_growth)]

    #     # Return the EBIT Multiple if a match is found, else return None
    #     if not result.empty:
    #         return result['EBIT Multiple'].values[0]
    #     else:
    #         return None
    
    def calculate_risk_of_dentist_leaving(self, dentist_contribution_table):
            
        df = dentist_contribution_table
        
        df['Contribution (%)'] = df['Sales Contribution ($)'] / df['Sales Contribution ($)'].sum()
        
        risk = df[df['Possibly Leaving?'] == True]['Contribution (%)'].sum()
        
        return risk
        
    def ebit_multiple_adjustment_due_dentist(self, ebit_multiple, risk):
        
        
        ebit_multiple = ebit_multiple * (1 - risk)
        
        
        return ebit_multiple
    
    def ebit_multiple_adjustment_due_net_sales_variation(self, ebit_multiple, relative_variability_net_sales):
        
        df_logic = pd.read_csv('net_sales_variability_assumption.csv')
        
        def get_adjustment(variability):
            for index, row in df_logic.iterrows():
                lower_bound = row['Lower Value']
                upper_bound = row['Upper Value']
                
                # Special case for the first row to handle zero inclusion
                if index == 0 and lower_bound <= variability < upper_bound:
                    return row['Adjustment to Multiple']
                # General case for all other rows
                elif lower_bound < variability <= upper_bound:
                    return row['Adjustment to Multiple']
            
            # Return None or a default value if no condition matches
            return None
        
        adjustment_to_multiple = get_adjustment(relative_variability_net_sales)
        ebit_multiple = ebit_multiple * adjustment_to_multiple if adjustment_to_multiple is not None else ebit_multiple
        
        return ebit_multiple
    
    
    def ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(self, ebit_multiple, number_of_active_patients, relative_variability_patient_spending):
        
        df_number_patient = pd.read_csv('number_patient_assumption.csv')
        df_patient_spending_variability = pd.read_csv('patient_spending_variability_assumption.csv')
        
        def get_adjustment(variability, df):
            for index, row in df.iterrows():
                lower_bound = row['Lower Value']
                upper_bound = row['Upper Value']
                
                # Special case for the first row to handle zero inclusion
                if index == 0 and lower_bound <= variability < upper_bound:
                    return row['Adjustment to Multiple']
                # General case for all other rows
                elif lower_bound < variability <= upper_bound:
                    return row['Adjustment to Multiple']
            
            # Return None or a default value if no condition matches
            return None
        
        adjustment_to_multiple_due_number_patient = get_adjustment(number_of_active_patients, df_number_patient)
        adjustment_to_multiple_due_patient_spending_variability = get_adjustment(relative_variability_patient_spending, df_patient_spending_variability)
        ebit_multiple = ebit_multiple * adjustment_to_multiple_due_number_patient * adjustment_to_multiple_due_patient_spending_variability
        
        return ebit_multiple
    
    
    
    
    # OUTPUT SUPPORTING FUNCTION
    
    def plot_net_sales_yearly(self):
        # Check if the net_sales_yearly DataFrame is not None
        if self.net_sales_yearly is None:
            return None

        # Sort the DataFrame by 'Year'
        sorted_data = self.net_sales_yearly.sort_values('Year')

        # Create the Plotly line chart
        fig = px.line(sorted_data, x='Year', y='Net Sales', 
                      title='Yearly Net Sales', 
                      labels={'Year': 'Year', 'Net Sales': 'Net Sales'},
                      markers=True)  # Enable markers

        # Customize the layout if needed (optional)
        fig.update_layout(xaxis_title='Year', yaxis_title='Net Sales')
        
        # update layout for xticks to only show integer
        fig.update_xaxes(tickmode='linear')

        # Return the Plotly figure object
        return fig
    
    
    def plot_net_sales_monthly(self):
        # Check if the net_sales_yearly DataFrame is not None
        if self.net_sales_monthly is None:
            return None

        # Sort the DataFrame by 'Year' and 'Month' if needed
        sorted_data = self.net_sales_monthly.sort_values(['Year', 'Month'])

        # Combine the 'Month' and 'Year' columns to create a new 'Month-Year' column for the x-axis
        sorted_data['Month-Year'] = sorted_data['Month'] + '-' + sorted_data['Year'].astype(str)

        # Create the Plotly line chart
        fig = px.line(sorted_data, x='Month-Year', y='Net Sales', 
                      title='Yearly Net Sales', 
                      labels={'Month-Year': 'Month-Year', 'Net Sales': 'Net Sales'},
                      markers=True)  # Enable markers

        # Customize the layout if needed (optional)
        fig.update_layout(xaxis_title='Month-Year', yaxis_title='Net Sales')

        # Ensure the x-axis labels show all the months
        fig.update_xaxes(tickmode='linear')

        # Return the Plotly figure object
        return fig