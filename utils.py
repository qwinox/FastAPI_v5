import base64
from tensorflow.keras.preprocessing import image
from keras import models
import numpy as np
from PIL import Image
from io import BytesIO



def preprocess_image(image_bytes):
    image_bytes = image_bytes.resize((28, 28))
    image_bytes = image_bytes.convert('L')
    image_bytes = image.img_to_array(image_bytes)
    image_bytes = image_bytes.reshape(784)              # Меняем форму массива в плоский вектор
    image_bytes = 255 - image_bytes                     # Инвертируем изображение
    image_bytes /= 255                                  # Нормализуем изображение

    return image_bytes

def get_result(image_file, is_api=False):

    image_bytes = image_file.file.read()
    encoded_string = base64.b64encode(image_bytes)
    bs64 = encoded_string.decode('utf-8')
    image_data = f'data:image/jpeg;base64,{bs64}'

    img = Image.open(BytesIO(image_bytes))
    img = preprocess_image(img)

    model = models.load_model('model/fashion_mnist_dense.h5')
    classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
    x = img
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    index = np.argmax(prediction)                           # находит индекс максимального элемента

    result = {
        "inference_time": str(round(prediction[0, index] * 100, 2)),
        "predictions": {
            "class_id": str(index),
            "class_name": str(classes[index])
        }
        }

    if not is_api:
        result["image_data"] = image_data

    return result