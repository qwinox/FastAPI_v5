from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_read_main():  # проверяет доступность приложения при обращении к корню сервер
    response = client.get("/")
    assert response.status_code == 200


def test_predict():  #
    file_name = 'static/images/test_image.jpg'
    response = client.post("/predict", files={"file": ("test_image", open(file_name, "rb"), "image/jpeg")})
    assert response.status_code == 200
