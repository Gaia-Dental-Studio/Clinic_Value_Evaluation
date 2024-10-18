import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def app():
    
    st.markdown("##### Definition")
    st.write("Risk of leaving dentist measures the impact of any possibility of existing dentist leaving the clinic in coming, near-future periods.")
    st.write("The impact itself is measured by the percentage of revenue that the dentist contributes to the clinic's total revenue.")
    
    st.markdown("##### Formula")
    st.image(r'variable_pages/risk_of_leaving_dentist_formula.png')
    st.write("For example if a particular dentist contributes 20% of the total revenue, the risk of leaving for that particular dentist is translated as expected reduction of revenue by 20%.")

    dataframe = pd.read_csv(r'variable_pages/dentist_contribution.csv')
    
    st.markdown('##### Baseline Assumption')
    
    st.dataframe(dataframe, hide_index=True)
    
    st.write("The baseline assumption is that the clinic has no dentist leaving in the near future. Above table shows example how the data to calculate the risk of leaving dentist is structured.")
    st.write("The example data above is belong to Acre Dental, a clinic that has 5 dentists. Each dentist contributes different percentage of revenue to the clinic.")