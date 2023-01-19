import pytest
import numpy as np

# add parent directory to path
import sys
sys.path.insert(0,'..')
from send_request import send_request

# TODO refactor the server such that it can take a dict of allowed usernames/passwords for better testing
def test_wrongUsernameDeniesAccess_and_raisesError():
    with pytest.raises(RuntimeError):
        send_request("should not run this import", "any_function()", username="foo", password="qQ32XALjF9JqFh!vF3xY")

def test_wrongPassWordDeniesAccess_and_raisesError():
    with pytest.raises(RuntimeError):
        send_request("should not run this import", "any_function()", username="tim9220", password="foo")

def test_correctUsernamePassword_runsRequestOnServer():
    result = send_request("import numpy as np", "np.linspace(0,10,20)", username="tim9220", password="qQ32XALjF9JqFh!vF3xY")
    assert result.all() == np.linspace(0,10,20).all()
    

