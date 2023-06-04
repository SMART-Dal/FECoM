import pytest
import numpy as np

from tool.measurement.send_request import send_request

# TODO refactor the server such that it can take a dict of allowed usernames/passwords for better testing
def test_wrongUsernameDeniesAccess_and_raisesError():
    with pytest.raises(RuntimeError):
        send_request("should not run this import", "any_function()", username="foo", password="qQ32XALjF9JqFh!vF3xY")

def test_wrongPassWordDeniesAccess_and_raisesError():
    with pytest.raises(RuntimeError):
         send_request("should not run this import", "any_function()", username="tim9220", password="foo")

def test_correctUsernamePassword_runsRequestOnServer():
    function_to_run = "np.linspace(0,10,20)"
    result =  send_request("import numpy as np", function_to_run, username="tim9220", password="qQ32XALjF9JqFh!vF3xY", return_result=True)
    assert result[function_to_run]["return"].all() == np.linspace(0,10,20).all()
    

