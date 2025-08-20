from _helpers.builders.base import Builder
from copy import deepcopy

class CircuitSubmitterRegistry:
    def __init__(self):
        self._submitters = {}
    
    def store_submitter(self, object, name):
        self._submitters[name] = object
    
    def get_submitter(self, name):
        return self._submitters.get(name)
    
    def list_submitters(self):
        return list(self._submitters.keys())

submitter_registry = CircuitSubmitterRegistry()

# The registry assumes hardware params and init_control_params remain constant for specific backends throughout runs!
class OptimiserRegistry:
    def __init__(self):
        self.backends = {}
    
    def store_optimisation(self, backend, rel_builder: Builder, opt_values):
        if not self.backends.get(backend):
            self.backends[backend] = {}

        opt_values_copy = deepcopy(opt_values)
        self.backends[backend][rel_builder] = opt_values
    
    def get_optimisation(self, backend, rel_builder: Builder):
        opts = self.backends.get(backend)
        
        if not opts:
            return None

        for cur_builder, opt in opts.items():
            if cur_builder is rel_builder:
                return opt
            
        return None
    
    def print_submitters(self):
        for backend, dicts in self.backends.items():
            print(f"Now listing optimisations for backend {backend}:")
            for cur_builder, opt in dicts.items():
                print(f"    {str(cur_builder)}: {opt}")

optimiser_registry = OptimiserRegistry()
