from dummy_clinic_generator import DummyClinicModel
import streamlit as st
import pandas as pd
import numpy as np

st.title("The Interface")



reference_clinic_df = pd.read_csv('reference_clinic_df.csv')
other_parameter_dict = {"mean_cogs": 0.34, "std_cogs": 0.198,
                        "mean_expenses": 0.42, "std_expenses": 0.24,
                        "mean_depreciation": 0.0339, "std_depreciation": 0.03175}


model = DummyClinicModel(reference_clinic_df, other_parameter_dict)

st.markdown("### Parameter Value Assumption")

col1, col2 = st.columns(2)

mean_revenue, std_revenue = model.dist_param(by='clinic_revenue')
mean_cogs, std_cogs = other_parameter_dict['mean_cogs'], other_parameter_dict['std_cogs']
mean_expenses, std_expenses = other_parameter_dict['mean_expenses'], other_parameter_dict['std_expenses']
mean_depreciation, std_depreciation = other_parameter_dict['mean_depreciation'], other_parameter_dict['std_depreciation']

with col1:
    st.metric("Mean Revenue", f"${mean_revenue:,.0f}")
    st.metric("Mean COGS (%)", f"{mean_cogs * 100:.2f}%")
    st.metric("Mean Expenses (%)", f"{mean_expenses * 100:.2f}%")
    st.metric("Mean Depreciation (%)", f"{mean_depreciation * 100:.2f}%")
with col2:
    st.metric("Standard Deviation Revenue", f"${std_revenue:,.0f}")
    st.metric("Standard Deviation COGS (%)", f"{std_cogs * 100:.2f}%")
    st.metric("Standard Deviation Expenses (%)", f"{std_expenses * 100:.2f}%")
    st.metric("Standard Deviation Depreciation (%)", f"{std_depreciation * 100:.2f}%")
    
st.markdown("### Generate Dummy Clinics")

number_of_clinics = st.slider("Number of Clinics", 1, 100, 10)
dummy_clinic_df = model.generate_dummy_clinic(number_of_clinics)

# create histogram using model method of revenue_histogram 
st.dataframe(dummy_clinic_df, hide_index=True)

st.markdown("### Histogram")

histogram_by = st.selectbox("Histogram by", ['clinic_revenue', 'clinic_cogs', 'clinic_expenses', 'clinic_depreciation'])

st.pyplot(model.histogram(dummy_clinic_df, by=histogram_by))





    