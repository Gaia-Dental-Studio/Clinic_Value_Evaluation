import pandas as pd
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
        self.tangible_assets = company_variables.get('Tangible Assets (PP&E)', 0)
        self.general_expense = self.operational_expense + self.other_expense
        self.ebitda = self.net_sales + self.cogs + self.trading_income + self.other_income + self.general_expense
        self.depreciation_total = self.clinic_depreciation + self.non_clinic_depreciation
        self.tax_total = self.bank_tax_expense + self.other_tax
        self.ebit = self.ebitda + self.depreciation_total + self.tax_total
        self.ebit_ratio = self.ebit / self.net_sales if self.net_sales > 0 else None
        self.net_sales_growth = company_variables.get('Net Sales Growth', 0)
        self.relative_variability_net_sales = company_variables.get('Relative Variation of Net Sales', 0)
        self.number_of_patients = company_variables.get('Number of Active Patients', 0)
        self.relative_variability_patient_spending = company_variables.get('Relative Variation of Patient Spending', 0)
        self.potential_existing_dentist_leaving = company_variables.get('Potential Existing Dentist Leaving', 0)
        self.number_of_dentist = company_variables.get('Number of Dentist', 0)
        self.projected_number_of_dentist = company_variables.get('Projected Number of Dentist', 0)
        

        # Extract DataFrames from the company variables dictionary
        self.net_cash_flow = company_variables.get('Net Cash Flow', None)
        self.equipment_life = company_variables.get('Equipment_Life', pd.DataFrame())

    def analyze_cash_flow(self):
        # Replace None values with 0 in the net_cash_flow DataFrame
        self.net_cash_flow = self.net_cash_flow.fillna(0)

        # Transform the wide-format net_cash_flow DataFrame into a long format
        long_data = self.net_cash_flow.reset_index().melt(id_vars='index', var_name='Month', value_name='Net Cashflow')
        long_data.rename(columns={'index': 'Year'}, inplace=True)
        long_data['Year'] = long_data['Year'].astype(int)

        # Sort data by Year and Month
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        long_data['Month'] = pd.Categorical(long_data['Month'], categories=month_order, ordered=True)
        long_data = long_data.sort_values(['Year', 'Month']).reset_index(drop=True)

        # Convert Net Cashflow values to numeric, forcing errors to NaN
        long_data['Net Cashflow'] = pd.to_numeric(long_data['Net Cashflow'], errors='coerce')

        # Drop rows with missing values (None or NaN)
        cleaned_data = long_data.dropna(subset=['Net Cashflow'])

        # Calculate the standard deviation of the Net Cashflow
        std_deviation = cleaned_data['Net Cashflow'].std()
        
        # Calculate the average of the Net Cashflow
        average_cashflow = cleaned_data['Net Cashflow'].mean()

        # Calculate the trend line coefficient (slope) using linear regression
        if not cleaned_data.empty:
            cleaned_data['Time_Index'] = np.arange(len(cleaned_data))  # Create a time index for the regression
            X = cleaned_data[['Time_Index']]
            y = cleaned_data['Net Cashflow']
            model = LinearRegression()
            model.fit(X, y)
            trend_coefficient = model.coef_[0]
        else:
            trend_coefficient = 0  # If there's no data, return 0 for the trend coefficient

        return average_cashflow, std_deviation, trend_coefficient
    
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
            self.tangible_assets = variable_dict.get('Tangible Assets (PP&E)', 0)
            self.general_expense = self.operational_expense + self.other_expense
            self.ebitda = self.net_sales + self.cogs + self.trading_income + self.other_income + self.general_expense
            self.depreciation_total = self.clinic_depreciation + self.non_clinic_depreciation
            self.tax_total = self.bank_tax_expense + self.other_tax
            self.ebit = self.ebitda + self.depreciation_total + self.tax_total
            self.ebit_ratio = self.ebit / self.net_sales if self.net_sales > 0 else None
            self.net_sales_growth = variable_dict.get('Net Sales Growth', 0)
            self.relative_variability_net_sales = variable_dict.get('Relative Variation of Net Sales', 0)
            self.number_of_patients = variable_dict.get('Number of Active Patients', 0)
            self.relative_variability_patient_spending = variable_dict.get('Relative Variation of Patient Spending', 0)
            self.potential_existing_dentist_leaving = variable_dict.get('Potential Existing Dentist Leaving', 0)
            self.number_of_dentist = variable_dict.get('Number of Dentist', 0)
            self.projected_number_of_dentist = variable_dict.get('Projected Number of Dentist', 0)

        # Read the net_cash_flow data from the 'net_cash_flow' sheet
        net_cash_flow_df = pd.read_excel(uploaded_file, sheet_name='net_cash_flow', index_col=0)

        # Ensure that the index (year) is properly formatted as integers
        net_cash_flow_df.index = net_cash_flow_df.index.astype(int)

        # Ensure that the DataFrame contains all the default years ['2019', '2020', '2021', '2022', '2023']
        default_years = [2019, 2020, 2021, 2022, 2023]
        for year in default_years:
            if year not in net_cash_flow_df.index:
                # Add missing years with None values for all months
                net_cash_flow_df.loc[year] = [None] * 12

        # Reorder the index to maintain the order of default years
        net_cash_flow_df = net_cash_flow_df.reindex(default_years)

        self.net_cash_flow = net_cash_flow_df


    def ebit_baseline_to_multiple(self, net_sales_growth):
        
        ebit = self.ebit
        ebit_ratio = self.ebit_ratio
        
        
        data = pd.read_csv('EBIT_baseline_to_multiple.csv')
        # Define boundary values based on the unique values in the dataset
        ebit_boundaries = sorted(data['EBIT'].unique())
        ebit_ratio_boundaries = sorted(data['EBIT Ratio'].unique())
        net_sales_growth_boundaries = sorted(data['Net Sales Growth'].unique())

        # Round down to nearest boundary value
        ebit = max([val for val in ebit_boundaries if val <= ebit])
        ebit_ratio = max([val for val in ebit_ratio_boundaries if val <= ebit_ratio])
        net_sales_growth = max([val for val in net_sales_growth_boundaries if val <= net_sales_growth])

        # Lookup the row with the closest match
        result = data[(data['EBIT'] == ebit) & 
                    (data['EBIT Ratio'] == ebit_ratio) & 
                    (data['Net Sales Growth'] == net_sales_growth)]

        # Return the EBIT Multiple if a match is found, else return None
        if not result.empty:
            return result['EBIT Multiple'].values[0]
        else:
            return None
        
    def ebit_multiple_adjustment_due_dentist(self, ebit_multiple, number_of_dentist, projected_number_of_dentist, possibility_existing_dentist_leaving):
        
        weight_coefficient = 0.6 if possibility_existing_dentist_leaving else 1
        
        if number_of_dentist <= projected_number_of_dentist:
            ebit_multiple = ebit_multiple * (projected_number_of_dentist / number_of_dentist * weight_coefficient)
        
        
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