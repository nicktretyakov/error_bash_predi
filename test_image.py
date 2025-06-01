
#!/usr/bin/env python3
from PIL import Image
import numpy as np

# Создаем простое тестовое изображение 32x32
img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save("test_image.jpg")
print("Тестовое изображение создано: test_image.jpg")
