import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class DummyClinicModel:
    def __init__(self, reference_clinic_df, other_parameter_dict):
        """
        Initialize the DummyClinicModel with a reference DataFrame and parameters.
        
        Parameters:
        - reference_clinic_df (pd.DataFrame): DataFrame containing clinic names and revenues.
        - other_parameter_dict (dict): Dictionary containing mean and std for various parameters.
        """
        self.reference_clinic_df = reference_clinic_df
        self.other_parameter_dict = other_parameter_dict

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
            mean = self.other_parameter_dict[f"mean_{by.split('_')[-1]}"]
            std_dev = self.other_parameter_dict[f"std_{by.split('_')[-1]}"]
        return mean, std_dev

    def generate_values(self, by, num_samples=1):
        """
        Generate random values for a given attribute based on its distribution,
        ensuring no negative values by re-sampling when necessary.
        
        Parameters:
        - by (str): The attribute to generate values for (e.g., 'clinic_revenue', 'clinic_cogs').
        - num_samples (int): Number of samples to generate.
        
        Returns:
        - np.ndarray: Array of generated values.
        """
        mean, std_dev = self.dist_param(by)
        values = []
        
        while len(values) < num_samples:
            sample = np.random.normal(mean, std_dev)
            if sample >= 0:
                values.append(sample)
        
        return np.array(values)

    def generate_dummy_clinic(self, number_of_clinics):
        """
        Generate a dummy clinic DataFrame with random values for multiple attributes.
        
        Parameters:
        - number_of_clinics (int): Number of dummy clinics to generate.
        
        Returns:
        - pd.DataFrame: DataFrame with columns for clinic names, revenues, COGS, expenses, and depreciation.
        """
        clinic_names = [f"Clinic {i+1}" for i in range(number_of_clinics)]
        clinic_revenues = np.round(self.generate_values('clinic_revenue', num_samples=number_of_clinics),0)
        clinic_cogs = np.round(self.generate_values('clinic_cogs', num_samples=number_of_clinics) * clinic_revenues,0)
        clinic_expenses = np.round(self.generate_values('clinic_expenses', num_samples=number_of_clinics) * clinic_revenues,0)
        clinic_depreciation = np.round(self.generate_values('clinic_depreciation', num_samples=number_of_clinics) * clinic_revenues,0)

        dummy_clinic_df = pd.DataFrame({
            'clinic_name': clinic_names,
            'clinic_revenue': clinic_revenues,
            'clinic_cogs': clinic_cogs,
            'clinic_expenses': clinic_expenses,
            'clinic_depreciation': clinic_depreciation
        })
        return dummy_clinic_df

    def histogram(self, dummy_clinic_df, by):
        """
        Create a histogram for a given attribute with x-axis formatted in thousands.
        
        Parameters:
        - dummy_clinic_df (pd.DataFrame): DataFrame containing clinic data.
        - by (str): The attribute to plot a histogram for (e.g., 'clinic_revenue', 'clinic_cogs').
        
        Returns:
        - matplotlib.figure.Figure: The histogram figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dummy_clinic_df[by], bins=10, color='blue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Histogram of {by.replace("_", " ").capitalize()}', fontsize=16)
        ax.set_xlabel(f'{by.replace("_", " ").capitalize()} (in Thousands)', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.grid(axis='y', alpha=0.75)

        # Format x-axis ticks in thousands
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}K"))
        return fig
