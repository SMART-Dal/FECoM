import dill as pickle

class FunctionDetails():
    """
    Store the information needed for running a function on the server.
    The method_object is pickled upon setting for cases where this object
    is a custom class. FunctionDetails objects are serialised and sent to
    the server, where they are deserialised. The double-pickled
    method_object will remain serialised until the getter is called such
    that the custom class definition can be loaded first onto the server.
    """
    def __init__(self, imports: str, function_to_run: str, args: list, kwargs: dict, max_wait_secs: int, wait_after_run_secs: int, return_result: bool, method_object: object, object_signature: str, custom_class: str, module_name: str, exec_not_eval: bool):
        # basic function details
        self.imports = imports
        self.function_to_run = function_to_run
        self.args = args
        self.kwargs = kwargs
        
        # configuration for running the function
        self.max_wait_secs = max_wait_secs
        self.wait_after_run_secs = wait_after_run_secs
        self.return_result = return_result # parameter for testing purposes
        self.exec_not_eval = exec_not_eval # execute function_to_run, do not evaluate (useful for passing more than just a single function call)
        
        # required for running a method on an object
        self.__method_object = pickle.dumps(method_object)
        self.object_signature = object_signature # This is used as a key(as in dictionary) for the object method calls response

        # required for running a method on an object of a custom class (TODO this could also be used for custom functions)
        self.custom_class = custom_class
        self.module_name = module_name
    
    @property
    def method_object(self):
        return pickle.loads(self.__method_object)


def build_function_details(imports: str, function_to_run: str, args: list = None, kwargs: dict = None, max_wait_secs: int = 0, wait_after_run_secs: int = 0, return_result: bool = False, method_object = None, object_signature = None, custom_class: str = None, exec_not_eval: bool = False) -> FunctionDetails:
    """
    Build a FunctionDetails object containg all the function data & settings for the server.
    """
    function_details = FunctionDetails(
        imports,
        function_to_run,
        args,
        kwargs,
        max_wait_secs,
        wait_after_run_secs,
        return_result,
        method_object,
        object_signature,
        custom_class,
        method_object.__module__ if custom_class is not None else None,
        exec_not_eval
    )
    return function_details