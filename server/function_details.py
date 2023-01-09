import pickle

class FunctionDetails():
    """
    Store the information needed for running a function on the server.
    The method_object is pickled upon setting for cases where this object
    is a custom class. FunctionDetails objects are serialised and sent to
    the server, where they are deserialised. The double-pickled
    method_object will remain serialised until the getter is called such
    that the custom class definition can be loaded first onto the server.
    """
    def __init__(self, imports: str, function_to_run: str, args: list = None, kwargs: dict = None, max_wait_secs: int = 0, method_object: object = None, custom_class: str = None, module_name: str = None):
        self.imports = imports
        self.function_to_run = function_to_run
        self.args = args
        self.kwargs = kwargs
        self.max_wait_secs = max_wait_secs
        self.__method_object = pickle.dumps(method_object)
        self.custom_class = custom_class
        self.module_name = module_name
    
    @property
    def method_object(self):
        return pickle.loads(self.__method_object)
    
    # TODO implement this for easier debugging
    # def __str__(self) -> str:
    #     pass