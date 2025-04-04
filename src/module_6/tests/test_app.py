from fastapi.testclient import TestClient
import pytest

from app import app, FeatureStore


client = TestClient(app)

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "200"}



def test_make_prediction_UserNotFound(monkeypatch: pytest.MonkeyPatch):
    user_id = "mock_user"

    payload = {"user_id": user_id}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 404
    assert response.json() == {"detail": f"User {user_id} not found"}


def test_make_prediction_PredictionException(monkeypatch: pytest.MonkeyPatch):

    def mock_get_features(self, user_id):
        return {'features':'mock_features'}
    
    monkeypatch.setattr(FeatureStore, "get_features", mock_get_features)

    response = client.post("/predict", json={"user_id": "fake_user"})
    
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction could not be made"}


