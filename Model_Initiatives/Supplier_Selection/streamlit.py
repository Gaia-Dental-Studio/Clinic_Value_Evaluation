import os
import sys
import streamlit as st
import pandas as pd
import requests


# Dynamically adjust sys.path to ensure imports work in any execution context
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the model
try:
    from Model_Initiatives.Supplier_Selection.model_supplier_selection import ModelSupplierSelection
except ModuleNotFoundError:
    # If running from the Supplier_Selection directory, adjust the import
    from model_supplier_selection import ModelSupplierSelection



st.title("Supplier Selection Model")

# Path handling for files
path_to_html = os.path.join(current_dir, "clinic_supplier_map.html")

try:
    with open(path_to_html, 'r') as f:
        html_data = f.read()
    st.markdown("### Clinics and Suppliers Map")
    st.components.v1.html(html_data, width=700, height=400)
except FileNotFoundError:
    st.error(f"File not found: {path_to_html}")

st.markdown("### Material Cost Matrix")

# Load CSV files
try:
    clinic_supplier_distance = pd.read_csv(
        os.path.join(current_dir, 'clinic_supplier_distance_matrix_single_clinic.csv'), index_col=0)
    price_matrix = pd.read_csv(
        os.path.join(current_dir, 'price_matrix_single_clinic.csv'), index_col=0)
    price_matrix = st.data_editor(price_matrix)
    demand_projection = pd.read_csv(
        os.path.join(current_dir, 'demand_projection_matrix_single_clinic.csv'), index_col=0)
    bom = pd.read_csv(os.path.join(current_dir, 'bom_single_clinic.csv'))
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
    
try:
    last_optimal = pd.read_csv('last_optimal_alloaction_single_clinic.csv')
except FileNotFoundError:
    last_optimal = None
    
# st.dataframe(demand_projection)

# Optimize button functionality
if st.button("Optimize"):
    progress_bar = st.progress(0)
    
    # Convert to JSON (orient='split' preserves DataFrame structure)
    data = {
        'clinic_supplier_distance': clinic_supplier_distance.to_json(orient='split'),
        'price_matrix': price_matrix.to_json(orient='split'),
        'demand_projection': demand_projection.to_json(orient='split'),
        'bom': bom.to_json(orient='split'),
        'last_optimal_allocation': last_optimal.to_json(orient='split') if last_optimal is not None else None
    }
    
    # save all the dataset to json file
    clinic_supplier_distance.to_json('json_file/clinic_supplier_distance.json', orient='split')
    price_matrix.to_json('json_file/price_matrix.json', orient='split')
    demand_projection.to_json('json_file/demand_projection.json', orient='split')
    bom.to_json('json_file/bom.json', orient='split')
    

    # st.write(demand_projection.to_json(orient='split'))


    # model = ModelSupplierSelection(clinic_supplier_distance, price_matrix, demand_projection, bom)
    progress_bar.progress(25)
    


    # allocation_results = model.solve('C:\\bin\\cbc.exe')
    
    progress_bar.progress(50)

    # Send POST request
    response = requests.post('http://127.0.0.1:5000/optimize', json=data)

    # Parse response
    result = response.json()
    
    # if last_optimal is not None:
    #     comparison = model.comparison_to_previous_optimal(last_optimal) 
    # else:
    #     comparison = model.comparison_to_previous_optimal() 
        
    # if result['Allocation'] is not None:
    #     result['Allocation'].to_csv('last_optimal_alloaction_single_clinic.csv', index=False)
        
    #     st.dataframe(result['Allocation'], use_container_width=True)
    
    progress_bar.progress(100)
    # cost_breakdown = model.get_cost_breakdown()
    # unit_cost_total = cost_breakdown['Unit Cost Total']
    # transportation_cost_total = cost_breakdown['Transportation Cost Total']
    # overall_cost_total = cost_breakdown['Overall Cost Total']
    
    st.write(result)

    if 'error' in result:
        st.error(f"Error: {result['error']}")
    else:
        st.write("Baseline Best Solution:", result['Baseline Best Solution'])
        st.write("New Solution Value:", result['New Solution Value'])
        st.write("Opportunity Gain:", result['Opportunity Gain'])

        if result['Allocation'] is not None:
            # Deserialize and ensure Clinic is a column
            allocation_df = pd.read_json(result['Allocation'], orient='split')
            print("Deserialized Allocation DataFrame:")
            print(allocation_df.head())

            # Ensure 'Clinic' is a column, not an index
            if allocation_df.index.name == 'Clinic':
                allocation_df.reset_index(inplace=True)

            # Display the allocation results
            st.dataframe(allocation_df, use_container_width=True)

        allocation_df.to_csv('last_optimal_alloaction_single_clinic.csv', index=False)

    # col1, col2, col3 = st.columns(3)

    # # with col1:
    # #     st.metric("Unit Cost Total", f"{unit_cost_total:,.0f}")

    # # with col2:
    # #     st.metric("Transportation Cost Total", f"{transportation_cost_total:,.0f}")

    # with col2:
    #     st.metric("Overall Cost Total", f"{overall_cost_total:,.0f}")
        
    
    # st.write(comparison if comparison is not None else "No comparison available")
    
