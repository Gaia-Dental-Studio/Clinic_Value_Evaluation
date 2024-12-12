import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import truncnorm


class DummyClinicModel:
    def __init__(self, reference_clinic_df, other_parameter_dict, item_code_df, demand_prob_item_code):
        """
        Initialize the DummyClinicModel with a reference DataFrame and parameters.
        
        Parameters:
        - reference_clinic_df (pd.DataFrame): DataFrame containing clinic names and revenues.
        - other_parameter_dict (dict): Dictionary containing mean and std for various parameters.
        """
        self.reference_clinic_df = reference_clinic_df
        self.other_parameter_dict = other_parameter_dict
        self.item_code_df = item_code_df
        self.demand_prob_item_code = demand_prob_item_code
        # self.demand_prob_item_code['Code'] = self.demand_prob_item_code['Code'].astype(str).str.zfill(3)
        

    def dist_param(self, by):
        """
        Get the mean and standard deviation for a given parameter.
        
        Parameters:
        - by (str): The attribute to calculate distribution parameters for (e.g., 'clinic_revenue').
        
        Returns:
        - tuple: (mean, standard deviation) of the specified attribute.
        """
        if by == 'clinic_revenue':
            values = self.reference_clinic_df['clinic_revenue'].values
            mean = np.mean(values)
            std_dev = np.std(values)
        else:
            key_suffix = "_".join(by.split("_")[1:])  # Extract everything after the first underscore
            mean = self.other_parameter_dict[f"mean_{key_suffix}"]
            std_dev = self.other_parameter_dict[f"std_{key_suffix}"]
        return mean, std_dev

    def generate_values(self, by, num_samples=1):
        """
        Generate random values for a given attribute based on a truncated normal distribution,
        covering the range mean - std to mean + std.

        Parameters:
        - by (str): The attribute to generate values for (e.g., 'clinic_revenue', 'clinic_cogs').
        - num_samples (int): Number of samples to generate.

        Returns:
        - np.ndarray: Array of generated values.
        """
        mean, std_dev = self.dist_param(by)
        
        # Define the lower and upper bounds for truncation
        lower_bound = mean - std_dev
        upper_bound = mean + std_dev
        
        # Calculate the normalized bounds for the truncnorm distribution
        a = (lower_bound - mean) / std_dev
        b = (upper_bound - mean) / std_dev
        
        # Generate samples from the truncated normal distribution
        values = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=num_samples)
        
        return values

    def generate_dummy_clinic(self, number_of_clinics):
        """
        Generate a dummy clinic DataFrame with random values for multiple attributes.
        
        Parameters:
        - number_of_clinics (int): Number of dummy clinics to generate.
        
        Returns:
        - pd.DataFrame: DataFrame with columns for clinic names, revenues, COGS, expenses, and depreciation.
        """
  
        
        clinic_names = self.reference_clinic_df['clinic_name'].tolist()
        clinic_revenues = np.round(self.generate_values('clinic_revenue', num_samples=number_of_clinics),0)
        clinic_cogs = np.round(self.generate_values('clinic_cogs_lab_material', num_samples=number_of_clinics) * clinic_revenues,0)
        clinic_expenses = np.round(self.generate_values('clinic_expenses', num_samples=number_of_clinics) * clinic_revenues,0)
        clinic_depreciation = np.round(self.generate_values('clinic_depreciation', num_samples=number_of_clinics) * clinic_revenues,0)

        dummy_clinic_df = pd.DataFrame({
            'clinic_name': clinic_names,
            'clinic_revenue': clinic_revenues,
            'clinic_cogs_lab_material': clinic_cogs,
            'clinic_expenses': clinic_expenses,
            'clinic_depreciation': clinic_depreciation
        })
        return dummy_clinic_df





    def histogram(self, dummy_clinic_df, by):
        """
        Create a histogram for a given attribute with x-axis formatted based on the attribute type,
        and include a vertical line indicating the mean of the data.
        
        Parameters:
        - dummy_clinic_df (pd.DataFrame): DataFrame containing clinic data.
        - by (str): The attribute to plot a histogram for (e.g., 'clinic_revenue', 'clinic_cogs_lab_material', 'clinic_ebit_ratio').
        
        Returns:
        - matplotlib.figure.Figure: The histogram figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(dummy_clinic_df[by], bins=10, color='blue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Histogram of {by.replace("_", " ").capitalize()}', fontsize=16)
        
        # Calculate mean
        mean_value = dummy_clinic_df[by].mean()
        
        # Add a vertical line for the mean
        ax.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
        ax.legend(fontsize=12)

        # Format the x-axis based on the attribute type
        if by == 'clinic_ebit_ratio':
            ax.set_xlabel(f'{by.replace("_", " ").capitalize()} (%)', fontsize=14)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        else:
            ax.set_xlabel(f'{by.replace("_", " ").capitalize()} (in Thousands)', fontsize=14)
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}K"))
        
        ax.set_ylabel('Frequency', fontsize=14)
        ax.grid(axis='y', alpha=0.75)
        
        return fig


    
    
    # def generate_item_code_demand(self, dummy_clinic_df):
        

    def generate_item_code_per_clinic(self, clinic_df):
        clarity_data = self.demand_prob_item_code
        item_code_df = self.item_code_df

        # Clean `item_number` column: remove invalid entries and ensure numeric values
        item_code_df = item_code_df[pd.to_numeric(item_code_df['item_number'], errors='coerce').notna()]
        item_code_df['item_number'] = item_code_df['item_number'].astype(int)

        # Precompute a price lookup dictionary for faster access
        price_lookup = {
            int(code): price
            for code, price in zip(item_code_df['item_number'], item_code_df['price AUD'])
        }
        
        cost_lookup = {
            int(code): cost
            for code, cost in zip(item_code_df['item_number'], item_code_df['cost_material AUD'])
        }

        # Convert `Code` and `demand_prob` columns to NumPy arrays for faster sampling
        clarity_codes = clarity_data['Code'].values
        clarity_probs = clarity_data['demand_prob'].values

        result_dict = {}  # To store results for each clinic

        for _, row in clinic_df.iterrows():
            clinic_name = row['clinic_name']
            clinic_revenue = row['clinic_revenue']

            revenue = 0
            selected_codes = []  # Store selected codes for counting

            while revenue < clinic_revenue:
                # Randomly select a Code based on demand_prob
                selected_code = np.random.choice(clarity_codes, p=clarity_probs)
                # Look up the price using the precomputed dictionary
                price_aud = price_lookup.get(selected_code, None)

                if price_aud == 0:
                    continue

                if price_aud is not None:
                    revenue += price_aud
                    selected_codes.append(selected_code)
                else:
                    # Skip if no price found (unlikely since we precomputed valid codes)
                    continue

            # Convert the selected codes into a DataFrame for grouping
            selected_df = pd.DataFrame({'Code': selected_codes})
            grouped_df = selected_df.groupby('Code').size().reset_index(name='Total Demand')

            # Add Total Revenue column
            grouped_df['Total Revenue'] = grouped_df['Code'].map(price_lookup) * grouped_df['Total Demand']
            grouped_df['Total Material Cost'] = grouped_df['Code'].map(cost_lookup) * grouped_df['Total Demand']

            # Keep only the required columns
            grouped_df = grouped_df[['Code', 'Total Demand', 'Total Revenue', 'Total Material Cost']]

            # Add the grouped DataFrame to the result_dict
            result_dict[clinic_name] = grouped_df

        return result_dict
    
    def calculate_cogs_salary(self, clinic_item_code, salary_OHT, salary_dentist, salary_specialist):
        
        item_code_df = self.item_code_df

        # Clean `item_number` column: remove invalid entries and ensure numeric values
        item_code_df = item_code_df[pd.to_numeric(item_code_df['item_number'], errors='coerce').notna()]
        item_code_df['item_number'] = item_code_df['item_number'].astype(int)
        dentist_quota = 60 * 8 * 325

        # Calculate Total Duration and Medical Officer for each code
        clinic_item_code['Total Duration'] = clinic_item_code['Total Demand'] * clinic_item_code['Code'].map(item_code_df.set_index('item_number')['duration'])
        clinic_item_code['Medical Officer'] = clinic_item_code['Code'].map(item_code_df.set_index('item_number')['medical_officer_new'])

        # Assign salaries based on Medical Officer type
        clinic_item_code['Total Salary'] = 0
        clinic_item_code.loc[clinic_item_code['Medical Officer'] == 'OHT', 'Total Salary'] = clinic_item_code['Total Duration'] / 60 * salary_dentist
        clinic_item_code.loc[clinic_item_code['Medical Officer'] == 'Specialist', 'Total Salary'] = clinic_item_code['Total Duration'] / 60 * salary_specialist
        # clinic_item_code.loc[clinic_item_code['Medical Officer'] == 'Dentist', 'Total Salary'] = clinic_item_code['Total Duration'] / 60 * salary_dentist

        # Calculate total duration for each type of Medical Officer
        total_duration_OHT = clinic_item_code[clinic_item_code['Medical Officer'] == 'OHT']['Total Duration'].sum()
        total_duration_specialist = clinic_item_code[clinic_item_code['Medical Officer'] == 'Specialist']['Total Duration'].sum()

        # Calculate number of staff needed
        number_OHT_staff = np.ceil((total_duration_OHT - dentist_quota) / (60 * 8 * 325)) if total_duration_OHT > dentist_quota else 0
        number_specialist_staff = np.ceil(total_duration_specialist / (60 * 8 * 325))

        # Adjust salary for OHT if exceeding dentist quota
        total_salary = clinic_item_code['Total Salary'].sum()
        if total_duration_OHT > dentist_quota:
            excess_duration_OHT = total_duration_OHT - dentist_quota
            total_salary -= excess_duration_OHT / 60 * salary_dentist
            total_salary += excess_duration_OHT / 60 * salary_OHT

        original_salary_OHT = clinic_item_code[clinic_item_code['Medical Officer'] == 'OHT']['Total Salary'].sum()
        new_salary_OHT = total_salary - clinic_item_code[clinic_item_code['Medical Officer'] == 'Specialist']['Total Salary'].sum()
        
        clinic_item_code.loc[clinic_item_code['Medical Officer'] == 'OHT', 'Total Salary'] *= new_salary_OHT / original_salary_OHT

        return number_OHT_staff, number_specialist_staff, total_salary, clinic_item_code

        

