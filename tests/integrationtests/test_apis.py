# import sys
# import os
# from fastapi.testclient import TestClient

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# sys.path.insert(0, project_root)

# from src.group40_leaf.api import app

# client = TestClient(app)

# def test_read_root():
#     response = client.get("/")
#     assert response.status_code == 200
#     assert response.json() == {"message": "Welcome to the Leaf model inference API!"}

import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    app = TestClient()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert b"Expected content" in response.data, "Expected content not found in response"