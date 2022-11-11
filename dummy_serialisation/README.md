# Np Array Serialisation Dummy App
Adapted from https://gist.github.com/andres-fr/f9c0d5993d7e7b36e838744291c26dde

We could use a similar approach for serialising data.

## Setup
Create a python virtual environment:  
``` python3 -m venv venv```   
Activate the venv (Linux):  
```source venv/bin/activate```  
Activate the venv (Windows):  
```venv\Scripts\activate```    
Install dependencies:  
```pip install flask numpy requests```

## Run
Start the client:  
```python3 client.py```  
Start the server in a separate terminal (you have to activate the venv here as well):  
```python3 server.py```  
Press return in the client terminal to send a numpy array to the server and receive the array multiplied by 10 as a respone.