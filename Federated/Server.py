from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import uvicorn
import jwt
import datetime


"""
Federated Learning Server API.

This module sets up a FastAPI-based federated learning server that handles:
- User authentication
- Global model distribution
- Collection of model updates from multiple clients
- Federated aggregation of model updates

Endpoints:
- `/global-model`: Retrieves the current global model.
- `/submit-update`: Receives local model updates from clients.
- `/aggregate`: Aggregates model updates using Federated Averaging.

Authentication:
- Uses JWT-based authentication to ensure secure access.

Dependencies:
- FastAPI for API handling
- NumPy for mathematical operations
- Pydantic for request validation

Usage:
1. Start the server using Uvicorn:
   `uvicorn Neuromah.Federated.Server:app --reload`
2. Clients interact with the server using `Client.py`.

"""



# Secret key for JWT encoding/decoding (store securely in production)
SECRET_KEY = "secret_mah"
ALGORITHM = "HS256"

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dummy in-memory storage for global model parameters (replace with persistent storage as needed)
global_model_parameters = None

# Data model for model updates (you might want to use a more complex structure)
class ModelUpdate(BaseModel):
    client_id: int
    parameters: dict  # You can refine this to match your parameters structure
    num_samples: int

# Dummy function to verify JWT tokens
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  # In production, verify expiry, audience, etc.
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

# Endpoint to generate token for demonstration purposes
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In a real implementation, verify the username/password here
    payload = {
        "sub": form_data.username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

# Endpoint to distribute the global model to clients
@app.get("/global-model")
def get_global_model(token: str = Depends(verify_token)):
    if global_model_parameters is None:
        raise HTTPException(status_code=404, detail="Global model not initialized")
    return {"global_model": global_model_parameters}

# Endpoint to receive model updates from clients
@app.post("/submit-update")
def submit_update(update: ModelUpdate, token: str = Depends(verify_token)):
    # Here, you would add the update to a queue or aggregate immediately.
    # For demonstration, we'll simply print and update a dummy global model.
    print(f"Received update from client {update.client_id} with {update.num_samples} samples")
    
    # TODO: Implement aggregation logic (e.g., FedAvg)
    # For demonstration, we simply override the global model parameters.
    global global_model_parameters
    if global_model_parameters is None:
        global_model_parameters = update.parameters
    else:
        # Here you would aggregate: weighted average between global_model_parameters and update.parameters
        # This is a placeholder: actual aggregation should use the number of samples for weighting.
        global_model_parameters = update.parameters  
    return {"detail": "Update received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
