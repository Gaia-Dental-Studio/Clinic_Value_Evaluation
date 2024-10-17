import streamlit as st
import pandas as pd
from model_for_strategy import *  # Importing the Model class from model.py
import numpy as np

def app():
    # Title and Description
    st.title("School Outreach Program")
    st.write("""
    School Outreach Program offers a dental health education program for schools as well as service for students and teachers. This initiative is designed to improve oral health, enhance dental health awareness, and generate additional revenue for the dental clinic.
    """)

    # Divider
    st.divider()

    # Layout using columns for a more compact design
    # with st.form("wellness_form"):
        
    st.markdown("#### Program Parameters")
    st.caption("Monthly Basis")
    # First row: Total Potential Employee and Conversion Rate
    col1, col2 = st.columns(2)
    with col1:
        total_schools = st.number_input("Total Schools", step=1, value=3)
        average_students = st.number_input("Average Students per School", step=1, value=500)
        total_students = st.number_input("Total Students", step=1, value=total_schools*average_students, disabled=True)
        
        
    with col2:
        
        conversion_rate = st.number_input("Conversion Rate (%)", step=1, value=10)
        event_frequency = st.number_input("Event Frequency", step=1, value=4, help="Assumed number of events executed to reach the specified conversion rate")
        converting_students = st.number_input("Converting Students", step=1, value=int(round(total_students * conversion_rate / 100)), disabled=True)
 
    # create dataframe with 2 columns: 'Number of Converting Parents' & 'Proportion (%)', and it will have rows as follows. For the first column: 0,1,2,3 and for the second column: 0.3, 0.5, 0.15, 0.05 respectively.
    converting_parents_df = st.data_editor(pd.DataFrame({'Number of Converting Parents': np.arange(4), 'Proportion (%)': [0.3, 0.5, 0.15, 0.05]}), hide_index=True)
 
    # calculate number of students with 'Number of Converting Parents' = 0 to see how many of converting students have no parents joining the program
    converting_students_no_parents = int(round(converting_students * converting_parents_df['Proportion (%)'][0]))
 
    col1, col2 = st.columns(2)
    
    with col1:
        
        # calculate total parents by calculate it this way: sumproduct of converting students * proportion (%) * number of converting parents, for all rows in converting_parents_df 
        # total_teachers_parents = st.number_input("Total Teachers & Parents", step=1, value=int(round(converting_students * converting_parents_df['Proportion (%)'].sum() * converting_parents_df['Number of Converting Parents']).sum()), disabled=True)
        
        total_converting_parents = 0
        
        for i in range(len(converting_parents_df)):
            total_converting_parents += converting_students * converting_parents_df['Proportion (%)'][i] * converting_parents_df['Number of Converting Parents'][i]
            # print(converting_parents_df['Proportion (%)'][i])
            
        
        converting_parents = st.number_input("Converting Parents", step=1, value=int(round(total_converting_parents)), disabled=True)
        total_converting = st.number_input("Total Converting/Joining", step=1, value=converting_students + converting_parents, disabled=True)
 
    with col2:

        discount_price_single = st.number_input("Discount Price Single (%)", step=5, value=10, help="Discount Price for Student if they join the program individually")
        discount_price_family = st.number_input("Discount Price Family (%)", step=5, value=20, help="Discount Price for Student if they join the program with their parents/family")
    
    converting_students_with_parents = total_converting - converting_students_no_parents
    
    discount_composition = st.data_editor(
        pd.DataFrame({
            'Discount Type': ['Single', 'Family'],
            'Discount (%)': [discount_price_single, discount_price_family],
            'Number Claimable': [converting_students_no_parents
                                 , converting_students_with_parents]
        }), 
        hide_index=True
    )
    
    
    package_list_df = pd.read_csv(r'school_outreach_data/package_list.csv')

    # Define all possible treatments for the multiselect options
    all_treatments = [
        'Consultation', 'Scaling', 'Periapical X-Ray', 'Filling', 'Dental Sealants',
        'Gum Surgery', 'Tooth Extraction', 'Bleaching', 'Crown', 'Bridge'
    ]


    # Display the editable table
    st.markdown('#### Package List')


    

    with st.expander("Modify Treatment Packages", expanded=False):
        
        st.write("Modify the Description field using the multiselect below:")

            # Create a function to handle the editing of the Description field with multiselect
        def edit_package_list(df):
            edited_df = df.copy()

            # Iterate through the rows of the dataframe
            for index, row in df.iterrows():
                # Multiselect option for the Description field
                selected_treatments = st.multiselect(
                    f"Edit Treatments for {row['Treatment Package']} - {row['Category']}",
                    all_treatments,
                    default=row['Description'].split(', ')
                )
                # Update the dataframe with the selected treatments
                edited_df.at[index, 'Description'] = ', '.join(selected_treatments)
            
            return edited_df

        # Get the edited dataframe
        edited_package_list_df = edit_package_list(package_list_df)
    
    package_list_df = st.data_editor(edited_package_list_df)


    st.markdown("#### Package Demand")
        
    
    model = ModelSchoolOutreach(converting_students, converting_parents, converting_students_no_parents, discount_price_single, discount_price_family)
        
    package_demand = model.calculate_package_demand(edited_package_list_df)
    # package_demand.to_csv('package_demand.csv', index=False)
    
    
    st.dataframe(package_demand.drop(columns=['Description']), hide_index=True)
        
    with st.expander("Treatment Prices & Cost", expanded=True):    
        edited_prices = st.data_editor(model.initial_price_df(), hide_index=True, 
                                    #    column_config={'Conversion Rate (%)': st.column_config.NumberColumn("Conversion Rate (%)", help="Conversion Rate out of Total Joined Program")}
                                       )
        
        # edited_prices.to_csv('edited_prices.csv', index=False)
        
        

    with st.expander("Event Cost", expanded = True):
        edited_event_cost = st.data_editor(model.event_cost_df, hide_index=True)
        
        # calculate total event cost by sumproduct, from edited_event_cost, column of 'Unit' and 'Cost per Unit (Rp.)'
        total_event_cost = (edited_event_cost['Unit'] * edited_event_cost['Cost per Unit (Rp.)']).sum()
        
        st.write(f"Total Event Cost (per Event): **Rp.{total_event_cost:,.0f}**")
        



    st.divider()


    st.markdown("#### Adjusted Prices")

    price_df = model.price_df(edited_prices)
    st.dataframe(price_df, hide_index=True)
    
    price_df.to_csv('price_df.csv', index=False)
    
    if st.button("Calculate"):
        
        st.header("Financial Performance (Monthly)")

        total_revenue, total_cost, total_profit = model.calculate_financials(price_df, package_demand, total_event_cost, event_frequency)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Revenue", f"Rp.{total_revenue:,.0f}")
            st.metric("Total Cost", f"Rp.{total_cost:,.0f}")


        with col2:
            st.metric("Total Profit", f"Rp.{total_profit:,.0f}")
            
        
        cashflow_df = GeneralModel().create_cashflow_df(total_revenue, total_cost, 1, 12, period_type='monthly')
        # st.dataframe(cashflow_df, hide_index=True)
        # st.write(f'Average Revenue: Rp.{cashflow_df["Revenue"].mean():,.0f}')
        # st.write(f'Total Revenue: Rp.{cashflow_df["Revenue"].sum():,.0f}')
        
        cashflow_df.to_csv('school_cashflow.csv', index=False)