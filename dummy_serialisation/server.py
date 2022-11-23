"""
Dummy server, interacts with the client receiving, altering and returning
compressed numpy arrays.
"""

from flask import Flask, request, Response
from config import SERVER_HOST, SERVER_PORT, API_PATH
from np_compressor import NPCompressor

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route(API_PATH, methods=['POST'])
def test1234():
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    r = request
    #
    data = NPCompressor.uncompress_nparr(r.data)
    #
    data10 = data*10
    print("\n\nReceived array (compressed size = "+\
          str(r.content_length)+"):\n"+str(data))
    resp, _, _ = NPCompressor.compress_nparr(data10)
    return Response(response=resp, status=200,
                    mimetype="application/octet_stream")


# start flask app
if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT)