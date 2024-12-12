from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# # Define your function to process multiple DataFrames
# def process_multiple_dfs(data_dict):
#     # Initialize the result dictionary
#     result = {}

#     # Iterate over the input dictionary
#     for idx, (initiative_name, df) in enumerate(data_dict.items()):
#         # Group by Quarter and sum Profit
#         profit_by_quarter = df.groupby('Quarter')['Profit'].sum().astype(int)

#         # Create the output dictionary for this initiative
#         result[idx] = {
#             'initiatives': initiative_name,
#             **profit_by_quarter.to_dict(),
            
#         }

#     return result

def process_multiple_dfs(filtered_clinic_projected_cashflow_set):
    result = {}

    for idx, (initiative_name, dataframes) in enumerate(filtered_clinic_projected_cashflow_set.items()):
        # Assuming 'dataframes' is a single DataFrame
        df = dataframes

        # Initialize the dictionary for this index
        result[idx] = {
            "initiatives": initiative_name,
            "horizon": {}
        }

        # Iterate through unique Horizon values
        for horizon in df["Horizon"].unique():
            # Filter the DataFrame for the current Horizon
            horizon_data = df[df["Horizon"] == horizon]

            # Add Horizon-specific data to the nested "horizon" dictionary
            result[idx]["horizon"][horizon] = {
                row["Quarter"]: int(row["Profit"]) for _, row in horizon_data.iterrows()
            }

    return result


# Define API endpoint
@app.route('/process_clinic_data', methods=['POST'])
def process_clinic_data():
    try:
        # Retrieve filtered_clinic_projected_cashflow_set from the request
        data = request.get_json()
        filtered_clinic_projected_cashflow_set = pickle.loads(bytes.fromhex(data['filtered_clinic_projected_cashflow_set']))

        # Process data to generate pool_clinic
        pool_clinic = {}
        for clinic_name, clinic_data in filtered_clinic_projected_cashflow_set.items():
            clinic_json = process_multiple_dfs(clinic_data)
            pool_clinic[clinic_name] = clinic_json

        # Return the processed pool_clinic as JSON
        return jsonify({'pool_clinic': pool_clinic})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
