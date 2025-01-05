import pandas as pd
import pyomo
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

class ModelSupplierSelection:
    def __init__(self, clinic_supplier_distance, price_matrix, demand_projection, bom, cost_per_unit_distance=0.0005):
        self.clinic_supplier_distance = clinic_supplier_distance
        self.price_matrix = price_matrix
        self.demand_projection = demand_projection
        self.bom = bom
        self.cost_per_unit_distance = cost_per_unit_distance

        self.material_demand = self._process_demand_projections()
        self.model = self._build_model()

    def _process_demand_projections(self):
        material_demand = pd.DataFrame()
        for clinic in self.demand_projection.index:
            clinic_demand = []
            for treatment, quantity in self.demand_projection.loc[clinic].items():
                treatment_materials = self.bom[self.bom['Code'] == treatment]
                for _, row in treatment_materials.iterrows():
                    material = row['Material']
                    required_quantity = row['Quantity'] * quantity
                    clinic_demand.append({'Clinic': clinic, 'Material': material, 'Demand': required_quantity})
            clinic_demand_df = pd.DataFrame(clinic_demand)
            material_demand = pd.concat([material_demand, clinic_demand_df], ignore_index=True)

        # Aggregate material demands per clinic
        return material_demand.groupby(['Clinic', 'Material'])['Demand'].sum().reset_index()

    def _build_model(self):
        model = pyo.ConcreteModel()

        # Sets
        model.clinics = pyo.Set(initialize=self.material_demand['Clinic'].unique())
        model.suppliers = pyo.Set(initialize=self.clinic_supplier_distance.columns)
        model.materials = pyo.Set(initialize=self.material_demand['Material'].unique())

        # Parameters
        # Demand (D_{i,k})
        material_demand_dict = self.material_demand.set_index(['Clinic', 'Material'])['Demand'].to_dict()
        model.demand = pyo.Param(model.clinics, model.materials, initialize=material_demand_dict, default=0)

        # Unit cost (C_{j,k})
        suppliers = self.price_matrix.columns.tolist()
        materials = self.price_matrix.index.tolist()
        unit_cost_dict_corrected = {
            (supplier, material): self.price_matrix.loc[material, supplier]
            for supplier in suppliers
            for material in materials
        }
        model.unit_cost = pyo.Param(model.suppliers, model.materials, initialize=unit_cost_dict_corrected, default=float('inf'))

        # Distance (T_{i,j})
        distance_dict = self.clinic_supplier_distance.stack().to_dict()
        model.distance = pyo.Param(model.clinics, model.suppliers, initialize=distance_dict, default=float('inf'))

        # Cost per unit distance
        model.cost_per_unit_distance = pyo.Param(initialize=self.cost_per_unit_distance)

        # Capacity (U_{j,k}) - Set to high values for now
        high_capacity = 1e6
        model.capacity = pyo.Param(model.suppliers, model.materials, initialize=lambda model, j, k: high_capacity)

        # Decision Variables
        model.x = pyo.Var(model.clinics, model.suppliers, model.materials, within=pyo.Binary)

        # Objective Function
        def objective_rule(model):
            return sum(
                model.x[i, j, k] * (
                    model.unit_cost[j, k] + model.cost_per_unit_distance * model.distance[i, j]
                ) * model.demand[i, k]
                for i in model.clinics for j in model.suppliers for k in model.materials
            )
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Constraints
        # Demand fulfillment
        def demand_fulfillment_rule(model, i, k):
            return sum(model.x[i, j, k] for j in model.suppliers) == 1
        model.demand_fulfillment = pyo.Constraint(model.clinics, model.materials, rule=demand_fulfillment_rule)

        # Capacity constraints
        def capacity_rule(model, j, k):
            return sum(model.x[i, j, k] * model.demand[i, k] for i in model.clinics) <= model.capacity[j, k]
        model.capacity_constraint = pyo.Constraint(model.suppliers, model.materials, rule=capacity_rule)

        return model

    def solve(self, solver_path='C:\\bin\\cbc.exe'):
        opt = SolverFactory('cbc', executable=solver_path) # note executable is not needed
        results = opt.solve(self.model, tee=True)

        # Extract results
        allocation = []
        for i in self.model.clinics:
            for j in self.model.suppliers:
                for k in self.model.materials:
                    if pyo.value(self.model.x[i, j, k]) > 0.5:
                        allocation.append({'Clinic': i, 'Supplier': j, 'Material': k})

        allocation_df = pd.DataFrame(allocation)
        
        # change the allocation_df column order to be 'Clinic', 'Material', 'Supplier'
        allocation_df = allocation_df[['Clinic', 'Material', 'Supplier']]
        
        
        allocation_df.to_csv('allocation_results.csv', index=False)
        return allocation_df

    def calculate_objective_value(self, allocation_df):
        # Calculate objective value for a given allocation
        total_cost = 0
        for _, row in allocation_df.iterrows():
            clinic = row['Clinic']
            supplier = row['Supplier']
            material = row['Material']

            # Filter material_demand directly instead of using query
            demand_value = self.material_demand[
                (self.material_demand['Clinic'] == clinic) & (self.material_demand['Material'] == material)
            ]['Demand'].values[0]

            # Calculate cost for this allocation
            total_cost += (
                self.price_matrix.loc[material, supplier] +
                self.cost_per_unit_distance * self.clinic_supplier_distance.loc[clinic, supplier]
            ) * demand_value
        return total_cost

    def comparison_to_previous_optimal(self, last_optimal_configuration=None):
        
        if last_optimal_configuration is None:
            new_allocation = self.solve()
            
            return {
                'Baseline Best Solution': None,
                'New Solution Value': self.calculate_objective_value(new_allocation),
                'Opportunity Gain': None,
                'Allocation': new_allocation}
                
        else:
        
            # Calculate baseline best solution objective value
            baseline_best_solution = self.calculate_objective_value(last_optimal_configuration)

            # Solve for the current optimal solution
            new_allocation = self.solve()
            new_solution_value = self.calculate_objective_value(new_allocation)

            # Calculate opportunity gain
            opportunity_gain = baseline_best_solution - new_solution_value


            
            return {
                'Baseline Best Solution': baseline_best_solution,
                'New Solution Value': new_solution_value,
                'Opportunity Gain': opportunity_gain,
                'Allocation': new_allocation if opportunity_gain > 0 else None
            }

    def get_cost_breakdown(self):
        unit_cost_total = sum(
            pyo.value(self.model.x[i, j, k]) * self.model.unit_cost[j, k] * self.model.demand[i, k]
            for i in self.model.clinics for j in self.model.suppliers for k in self.model.materials
        )

        transportation_cost_total = sum(
            pyo.value(self.model.x[i, j, k]) * self.model.cost_per_unit_distance * self.model.distance[i, j] * self.model.demand[i, k]
            for i in self.model.clinics for j in self.model.suppliers for k in self.model.materials
        )

        overall_cost_total = pyo.value(self.model.objective)

        return {
            'Unit Cost Total': unit_cost_total,
            'Transportation Cost Total': transportation_cost_total,
            'Overall Cost Total': overall_cost_total
        }
        
    def summarize_allocation(self, allocation_df):
        """
        Summarizes the allocation configuration into a human-readable paragraph for each clinic.

        Parameters:
            allocation_df (pd.DataFrame): A DataFrame with columns 'Clinic', 'Material', 'Supplier'.

        Returns:
            str: A formatted summary of the allocation configuration.
        """


        # Validate input DataFrame
        if not set(['Clinic', 'Material', 'Supplier']).issubset(allocation_df.columns):
            raise ValueError("The DataFrame must contain 'Clinic', 'Material', and 'Supplier' columns.")

        # Group by Clinic and Supplier and aggregate materials
        grouped = (
            allocation_df.groupby(['Clinic', 'Supplier'])
            .agg({'Material': lambda x: ', '.join(sorted(x))})
            .reset_index()
        )

        # Initialize the summary string
        summary = ""

        # Iterate through each clinic and create its paragraph
        for clinic in grouped['Clinic'].unique():
            clinic_df = grouped[grouped['Clinic'] == clinic]
            allocations = []

            for _, row in clinic_df.iterrows():
                allocations.append(f"Supplier {row['Supplier']} for Material {row['Material']}")

            clinic_summary = f"For Clinic {clinic}, we need to do procurement using {', '.join(allocations)}."
            summary += clinic_summary + "\n\n"

        # Return the summary, stripping the final newline
        return summary.strip()


