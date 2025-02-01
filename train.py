# train.py
import tensorflow as tf

# بارگذاری داده‌ها
from preprocessing import train_data, val_data
from tensorflow.keras.models import load_model

# بارگذاری مدل
model = load_model('cnn_model.h5')

# کامپایل مدل
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# آموزش مدل
history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=10,
                    batch_size=32)

# ذخیره مدل آموزش‌دیده
model.save('cnn_model_trained.h5')
print("Training Complete and Model Saved!")
