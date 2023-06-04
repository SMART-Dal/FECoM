"""
SERVER URL, PORT & DEBUG CONFIG
"""
SERVER_MODULE = "flask_server.py"

### SET TO FALSE FOR DEV SETTINGS ON YOUR LOCAL MACHINE ###
PROD = True

# server settings for localhost environment
DEV_HOST = "localhost"
DEV_PORT = 54321

# server settings for production environment
PROD_HOST = "129.173.67.60" # can also set to "0.0.0.0" on the server
PROD_PORT = 8080

# configure server settings here
API_PATH = "/api/run_experiment"
SERVER_HOST = PROD_HOST if PROD else DEV_HOST
SERVER_PORT = PROD_PORT if PROD else DEV_PORT
URL = "https://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH

TEMP_EXEC_CODE_FILE = Path("out/code_file_tmp.py")

"""
AUTHENTICATION & SSL CONFIG
"""
# authentication
USERS = {
    "tim9220": "pbkdf2:sha256:260000$HCuie8IPjFLW3ZcA$57da68fec635caf7922a77f64781669c5427ace7eae1d8cb67e218f5e363956f",
    "saurabh3188": "pbkdf2:sha256:260000$NEviyP0Zrg5kobkO$7e3037a9053816e51e0e2b920c92b76ae0b9708f2ca755eb48fe76fd07cb5fa4"
}

# HTTPS certificate relative paths from server directory
CA_CERT_PATH = Path("certificates/cert.pem") # public key
CA_KEY_PATH = Path("certificates/key.pem") # private key