from flask import Flask, request, jsonify
import pandas as pd
from model_supplier_selection import ModelSupplierSelection  # Ensure the model is accessible here
import json

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize():
    # try:
        # Parse input files
        data = request.get_json()
        
        # print("Data received:")
        # print(data)
        # print("End")

        # Deserialize data into DataFrames
        clinic_supplier_distance = pd.read_json(data['clinic_supplier_distance'], orient='split')
        # print(clinic_supplier_distance)
        price_matrix = pd.read_json(data['price_matrix'], orient='split')
        # print(price_matrix)
        demand_projection = pd.read_json(data['demand_projection'], orient='split')
        # print(demand_projection)
        bom = pd.read_json(data['bom'], orient='split')
        # print(bom)

        # Ensure proper index and column structure for demand_projection
        if demand_projection.index.name != 'Clinic':
            demand_projection.set_index('Clinic', inplace=True)
        print("Demand Projection (after processing):")
        print(demand_projection.head())

        # Normalize BoM
        bom['Code'] = bom['Code'].astype(str).str.strip()
        print("BoM (after processing):")
        print(bom.head())

        # Validate relationships between demand_projection and bom
        missing_treatments = [col for col in demand_projection.columns if col not in bom['Code'].unique()]
        if missing_treatments:
            return jsonify({'error': f"Missing treatments in BoM: {missing_treatments}"}), 400

        # Optional input
        last_optimal_allocation = None
        if 'last_optimal_allocation' in data and data['last_optimal_allocation']:
            last_optimal_allocation = pd.read_json(data['last_optimal_allocation'], orient='split')

        # Initialize and run the model
        model = ModelSupplierSelection(
            clinic_supplier_distance, price_matrix, demand_projection, bom
        )
        
        if last_optimal_allocation is not None:
        
            comparison = model.comparison_to_previous_optimal(last_optimal_allocation)
            
            print("PREVIOUS OPTIMAL ALLOCATION NOT NONE")
        
        else:
            comparison = model.comparison_to_previous_optimal()
            print("PREVIOUS OPTIMAL ALLOCATION NONE")

        # Serialize allocation if present
        if comparison['Allocation'] is not None:
            
            summary_allocation = model.summarize_allocation(comparison['Allocation'])
            if not isinstance(summary_allocation, str):
           
                summary_allocation = json.dumps(summary_allocation)  # Ensure it's a JSON string
 

            comparison['Allocation'] = comparison['Allocation'].to_json(orient='split')
            
        else:
            summary_allocation = None  # Default value if no allocation       
            

        # Return results
        return jsonify({
            'Baseline Best Solution': comparison['Baseline Best Solution'],
            'New Solution Value': comparison['New Solution Value'],
            'Opportunity Gain': comparison['Opportunity Gain'],
            'Allocation': comparison['Allocation'],
            'Summary Allocation': summary_allocation
        })

    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
