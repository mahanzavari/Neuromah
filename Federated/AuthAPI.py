import jwt
import datetime
from fastapi import HTTPException

SECRET_KEY = "neuromah"

class AuthAPI:
    """
Authentication API for Federated Learning.

This module handles user authentication using JSON Web Tokens (JWT).
It provides:
- User login functionality
- Secure token generation
- Token verification for API access
- Token management for multiple users

Features:
- Uses HS256 encryption for token security.
- Validates token expiration and integrity.
- Provides helper functions for client authentication.

Endpoints Used:
- `/login`: Generates authentication tokens for users.

Dependencies:
- JWT for secure token encoding/decoding.
- FastAPI HTTPException for error handling.

Example Usage:
```python
auth = AuthAPI()
token = auth.login(username="client1", password="securepass")
print(auth.verify_token(token))
"""
    def __init__(self, server_url=None):
        self.server_url = server_url
        self.tokens = {}

    def login(self, username, password):
        if username == "client1" and password == "securepass":
            token = jwt.encode(
                {"sub": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
                SECRET_KEY,
                algorithm="HS256"
            )
            self.tokens[username] = token
            return token
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return payload["sub"]
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def get_headers(self):
        return {"Authorization": f"Bearer {list(self.tokens.values())[0]}"}
