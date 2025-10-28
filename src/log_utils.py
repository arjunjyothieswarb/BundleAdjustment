from datetime import datetime
from termcolor import colored

def info(message: str) -> None:
    # Getting the time header
    curr_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    timeheader = "[%s]"%curr_time
    
    # Constructing the header
    header = "[INFO]" + timeheader + ": "
    
    # Printing the message
    print(header + message)

def warn(message: str) -> None:
    # Getting the time header
    curr_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    timeheader = "[%s]"%curr_time
    
    # Constructing the header
    header = "[WARNING]" + timeheader + ": "
    
    # Printing the message
    print(colored(header, "yellow") + message)

def error(message: str) -> None:
    # Getting the time header
    curr_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    timeheader = "[%s]"%curr_time
    
    # Constructing the header
    header = "[ERROR]" + timeheader + ": "
    
    # Printing the message
    print(colored(header, "red") + message)

if __name__ == "__main__":
    error("Hello!")