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

class ControlParameterRegistry():

    def __init__(self):
        self.control_parameters = {}

    def get_control_parameters(self):
        return self.control_parameters

    def set_control_parameters(self, control_params: dict):
        self.control_parameters = control_params

control_parameter_registry = ControlParameterRegistry()
