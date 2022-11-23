"""
Dummy client, interacts with the server sending and receiving
compressed numpy arrays.
Run:
python client.py
"""

import numpy as np
import requests
from config import SERVER_HOST, SERVER_PORT, API_PATH
from np_compressor import NPCompressor


# ## MAIN CLIENT ROUTINE

url = "http://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH
while True:
    input("\n\npress return...")
    arr = np.random.rand(3,3)
    compressed, u_sz, c_sz = NPCompressor.compress_nparr(arr)
    #
    print("\nsending array to", url)
    print("size in bits (orig, compressed):", u_sz, c_sz)
    print(arr)
    #
    resp = requests.post(url, data=compressed,
                         headers={'Content-Type': 'application/octet-stream'})
    #
    print("\nresponse:")
    data = NPCompressor.uncompress_nparr(resp.content)
    print(data)