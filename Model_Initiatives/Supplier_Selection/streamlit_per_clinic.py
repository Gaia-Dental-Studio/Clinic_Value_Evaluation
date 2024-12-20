import os
import sys
import streamlit as st
import pandas as pd
import pickle


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


def app(dataset_index, clinic_name):
    st.title("Clinic's Supplier Selection Model")
    


    dataset_dir = os.path.join("dummy_clinic_model/pkl_files", f"dataset_{dataset_index}")

    # Check if the directory exists
    if os.path.exists(dataset_dir):


        # Load the item codes per clinic dictionary
        with open(os.path.join(dataset_dir, 'item_code_per_clinic.pkl'), 'rb') as f:
            item_code_per_clinic = pickle.load(f)


        st.success(f"Loaded data from {dataset_dir}")
    else:
        st.error(f"The directory {dataset_dir} does not exist!")

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
            os.path.join(current_dir, 'clinic_supplier_distance_matrix.csv'), index_col=0)
        price_matrix = pd.read_csv(
            os.path.join(current_dir, 'price_matrix.csv'), index_col=0) # the best one, not the before
        price_matrix = st.data_editor(price_matrix)
        # demand_projection = pd.read_csv(
        #     os.path.join(current_dir, 'demand_projection_matrix.csv'), index_col=0)
        bom = pd.read_csv(os.path.join(current_dir, 'bom.csv'))
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
   
    item_code_clinic = item_code_per_clinic[clinic_name]
        
    item_code_demand = item_code_clinic[['Code', 'Total Demand']]
    item_code_demand["Code"] = item_code_demand["Code"].astype(str).apply(lambda x: x.zfill(3))
    item_code_demand['Clinic'] = clinic_name
    
    previous_overall_total_cost = item_code_clinic['Total Material Cost'].sum()

    # Initialize an empty list to store intermediate data
    data_list = []

    # # Loop through the dictionary to extract data
    # for clinic, df in yearly_demand_projection.items():
    #     for _, row in df.iterrows():
    #         data_list.append({"Clinic": clinic, "Code": row["Code"], "Total Demand": row["Total Demand"]})

    # # Convert the data list into a DataFrame
    # long_df = pd.DataFrame(data_list)

    # Pivot the DataFrame to create the demand projection matrix
    demand_projection_matrix = item_code_demand.pivot(index="Clinic", columns="Code", values="Total Demand")

    # Fill NaN with 0 if there are missing demand values
    demand_projection = demand_projection_matrix.fillna(0)
        
        
    clinic_supplier_distance = clinic_supplier_distance.loc[[clinic_name]]

    # Optimize button functionality
    if st.button("Optimize"):
        progress_bar = st.progress(0)

        model = ModelSupplierSelection(clinic_supplier_distance, price_matrix, demand_projection, bom)
        progress_bar.progress(25)

        allocation_results = model.solve('C:\\bin\\cbc.exe')
        progress_bar.progress(75)

        cost_breakdown = model.get_cost_breakdown()
        progress_bar.progress(100)

        unit_cost_total = cost_breakdown['Unit Cost Total']
        transportation_cost_total = cost_breakdown['Transportation Cost Total']
        overall_cost_total = cost_breakdown['Overall Cost Total']

        st.dataframe(allocation_results, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Unit Cost Total", f"{unit_cost_total:,.0f}")

        with col2:
            st.metric("Transportation Cost Total", f"{transportation_cost_total:,.0f}")

        with col3:
            st.metric("Overall Cost Total", f"{overall_cost_total:,.0f}")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Previous Overall Cost Total", f"{previous_overall_total_cost:,.0f}")
            
        with col2:
            st.metric("Cost Savings", f"{previous_overall_total_cost - overall_cost_total:,.0f}")
            
        with col3:
            cost_saving_percentage = (previous_overall_total_cost - overall_cost_total) / previous_overall_total_cost
            st.metric("Cost Saving Percentage", f"{cost_saving_percentage:.2%}")


        # save previous_overall_total_cost - overall_cost_total as pkl
        with open(os.path.join(current_dir, 'reduced_material_cost.pkl'), 'wb') as f:
            pickle.dump(previous_overall_total_cost - overall_cost_total, f)