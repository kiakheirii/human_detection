# preprocessing.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# تعریف مسیر داده‌ها
train_dir = "C:\\Users\\noavar\\Desktop\\uni_project\\output_dataset\\train"
val_dir = "C:\\Users\\noavar\\Desktop\\uni_project\\output_dataset\\val"
test_dir = r"C:\Users\noavar\Desktop\uni_project\output_dataset\test"

# پیش‌پردازش و افزایش داده
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# لود داده‌ها
train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

print("Preprocessing Done!")
