from datetime import datetime

def custom_print(location: str, message: str):
    """
    Will print a message with a timestampe in the form
    [LOCATION] [hh:mm:ss]
    """
    time_stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{location.upper()}] [{time_stamp}] " + message)