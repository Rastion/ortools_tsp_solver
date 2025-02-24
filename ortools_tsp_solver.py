from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from qubots.base_optimizer import BaseOptimizer

class ORToolsTSPSolver(BaseOptimizer):
    """
    OR‑Tools Routing-based TSP solver.

    This solver uses OR‑Tools’ Routing library to solve the TSP. It creates a single-vehicle
    routing model over the cities (with depot 0), uses the distance matrix from the TSP problem,
    and extracts the tour from the solution.
    """
    def __init__(self, time_limit=300):
        self.time_limit = time_limit

    def optimize(self, problem, initial_solution=None, **kwargs):
        # Number of cities
        n = problem.nb_cities

        # Create the routing index manager and routing model.
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # one vehicle, depot at city 0
        routing = pywrapcp.RoutingModel(manager)

        # Get the distance matrix from the problem.
        dist_matrix = problem.dist_matrix

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            # Convert from routing variable Index to city index.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Set search parameters.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        search_parameters.time_limit.seconds = self.time_limit

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            # Extract the tour.
            index = routing.Start(0)
            tour = []
            while not routing.IsEnd(index):
                tour.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            # Optionally, you can rotate the tour so that city 0 is first.
            if 0 in tour:
                idx = tour.index(0)
                tour = tour[idx:] + tour[:idx]
            cost = solution.ObjectiveValue()
            return tour, cost
        else:
            # If no solution is found, return a random solution with an infinite cost.
            return problem.random_solution(), float('inf')
