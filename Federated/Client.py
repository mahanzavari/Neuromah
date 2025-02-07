import requests

class FederatedClient:
    def __init__(self, server_url, username, password):
        self.server_url = server_url
        self.token = None
        self.authenticate(username, password)

    def authenticate(self, username, password):
        """Authenticate and store the token."""
        response = requests.post(f"{self.server_url}/login", json={"username": username, "password": password})
        if response.status_code == 200:
            self.token = response.json().get("token")
        else:
            raise Exception("Authentication failed")

    def get_global_model(self):
        """Retrieve the global model from the server."""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{self.server_url}/get-global-model", headers=headers)
        return response.json() if response.status_code == 200 else None

    def send_local_updates(self, local_model):
        """Send local model updates to the server."""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(f"{self.server_url}/send-local-updates", json=local_model, headers=headers)
        return response.json() if response.status_code == 200 else None
