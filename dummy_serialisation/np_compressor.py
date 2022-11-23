import io
import zlib
import numpy as np


class NPCompressor():
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
        Decompresses the given bytestring and converts it to a numpy array
        """
        return np.load(io.BytesIO(zlib.decompress(bytestring)))