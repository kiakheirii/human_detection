# predict.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# بارگذاری مدل
model = load_model('cnn_model_trained.h5')

# لود عکس جدید
img_path = r"C:\Users\noavar\Desktop\uni_project\labeled_images\0\frame_7391.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# پیش‌بینی
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
if predicted_class == 1:
    print("There is human in picture")
else :
    print("There is no human in picture")
