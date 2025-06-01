
#!/usr/bin/env python3

import requests
import json
import numpy as np
from PIL import Image
import sys

def image_to_array(image_path, size=(128, 128)):
    """Convert image to normalized array for API"""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(size)

        # Convert to normalized array
        img_array = np.array(img) / 255.0
        # Reshape to match expected format [C, H, W] -> [C*H*W]
        img_flat = img_array.transpose(2, 0, 1).flatten().tolist()

        return img_flat
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None

def test_os_error_prediction(image_path, server_url="http://localhost:5000"):
    """Test OS error prediction API"""

    print(f"Загрузка и обработка изображения: {image_path}")
    image_data = image_to_array(image_path)

    if image_data is None:
        return

    # Prepare request
    payload = {
        "image": image_data
    }

    try:
        print(f"Отправка запроса на {server_url}/predict-os-error...")
        response = requests.post(f"{server_url}/predict-os-error",
                               json=payload,
                               headers={'Content-Type': 'application/json'},
                               timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("\n=== Результат анализа ошибки ОС ===")
            print(f"Тип ошибки: {result['error_type']}")
            print(f"Операционная система: {result['os_type']}")
            print(f"Уверенность: {result['confidence']:.2%}")
            print(f"Описание: {result['description']}")
        else:
            print(f"Ошибка API: {response.status_code}")
            print(f"Ответ: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка подключения: {e}")
        print("Убедитесь, что сервер запущен: cargo run server")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python3 test_os_error_api.py <путь_к_скриншоту>")
        print("Пример: python3 test_os_error_api.py error_screenshot.png")
        sys.exit(1)

    image_path = sys.argv[1]
    test_os_error_prediction(image_path)
