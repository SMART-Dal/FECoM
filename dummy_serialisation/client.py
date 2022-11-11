################################################################################
### client.py
################################################################################


#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Dummy client, interacts with the server sending and receiving
compressed numpy arrays.
Run:
python client.py
"""


from __future__ import print_function
import io
import numpy as np
import zlib
import requests


# ## CONFIG

SERVER_HOST= "localhost"
SERVER_PORT = 12345
API_PATH = "/api/test"


# ## HELPERS

def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))


# ## MAIN CLIENT ROUTINE

url = "http://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH
while True:
    input("\n\npress return...")
    arr = np.random.rand(3,3)
    compressed, u_sz, c_sz = compress_nparr(arr)
    #
    print("\nsending array to", url)
    print("size in bits (orig, compressed):", u_sz, c_sz)
    print(arr)
    #
    resp = requests.post(url, data=compressed,
                         headers={'Content-Type': 'application/octet-stream'})
    #
    print("\nresponse:")
    data = uncompress_nparr(resp.content)
    print(data)