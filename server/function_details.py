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
    def __init__(self, imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs=0, method_object=None, custom_class: str = None, module_name: str = None):
        self.imports = imports
        self.custom_class = custom_class
        self.module_name = module_name
        self.function_to_run = function_to_run
        self.method_object = method_object
        self.args = function_args
        self.kwargs = function_kwargs
        self.max_wait_secs = max_wait_secs
    
    @property
    def method_object(self):
        return pickle.loads(self.__method_object)
    
    @method_object.setter
    def method_object(self, obj):
        self.__method_object = pickle.dumps(obj)