
#!/usr/bin/env python3

import asyncio
import websockets
import json
import base64
import sys

async def test_chat():
    uri = "ws://localhost:5000/ws/"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Подключен к чату с AI")

            # Тестовые сообщения
            test_messages = [
                {"message": "Привет! Можешь помочь с анализом ошибок?", "image_data": None},
                {"message": "Что такое BSOD?", "image_data": None},
                {"message": "Как исправить kernel panic в Linux?", "image_data": None},
                {"message": "помощь", "image_data": None},
            ]

            for msg in test_messages:
                print(f"\n📤 Отправка: {msg['message']}")
                await websocket.send(json.dumps(msg))

                response = await websocket.recv()
                data = json.loads(response)

                print(f"🤖 Ответ: {data['response']}")

                if data.get('analysis'):
                    analysis = data['analysis']
                    print(f"   📊 Анализ: {analysis['error_type']} ({analysis['os_type']})")
                    print(f"   🎯 Уверенность: {analysis['confidence']:.2%}")

                if data.get('suggestions'):
                    print(f"   💡 Рекомендации: {len(data['suggestions'])} пунктов")

                await asyncio.sleep(1)

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("Убедитесь, что сервер запущен: cargo run server")

def test_image_upload():
    """Тест загрузки изображения через base64"""
    try:
        # Создаем тестовое изображение 1x1 пиксель
        import io
        from PIL import Image

        # Создать простое тестовое изображение
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()

        # Кодировать в base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        print(f"✅ Тестовое изображение создано (размер: {len(img_base64)} символов)")
        return img_base64

    except ImportError:
        print("❌ Для тестирования изображений установите Pillow: pip install Pillow")
        return None

async def test_image_analysis():
    """Тест анализа изображения"""
    img_data = test_image_upload()
    if not img_data:
        return

    uri = "ws://localhost:5000/ws/"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Подключен для тестирования анализа изображений")

            message = {
                "message": "Проанализируй этот скриншот ошибки",
                "image_data": img_data
            }

            print("📤 Отправка тестового изображения...")
            await websocket.send(json.dumps(message))

            response = await websocket.recv()
            data = json.loads(response)

            print(f"🤖 Ответ: {data['response']}")

            if data.get('analysis'):
                analysis = data['analysis']
                print(f"📊 Результат анализа:")
                print(f"   Тип ошибки: {analysis['error_type']}")
                print(f"   ОС: {analysis['os_type']}")
                print(f"   Уверенность: {analysis['confidence']:.2%}")
                print(f"   Решения: {len(analysis['solutions'])}")

    except Exception as e:
        print(f"❌ Ошибка при тестировании изображения: {e}")

if __name__ == "__main__":
    print("🧪 Тестирование чата с AI помощником")
    print("="*50)

    if len(sys.argv) > 1 and sys.argv[1] == "image":
        print("Тестирование анализа изображений...")
        asyncio.run(test_image_analysis())
    else:
        print("Тестирование текстового чата...")
        asyncio.run(test_chat())

    print("\n✅ Тестирование завершено")
    print("Для тестирования изображений: python3 test_chat_api.py image")
