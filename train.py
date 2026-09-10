# train_model.py
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from PIL import ImageFile

# === Prevent crash on broken/corrupted images ===
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Dataset path ===
DATASET_DIR = r'C:\Users\BusinessComputers.in\Desktop\food project\dataset food'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # you can increase if training runs well

# === Data generators with validation split ===
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# === Training and Validation Generators ===
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Build model using Transfer Learning (MobileNetV2) ===
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False  # freeze the convolutional base

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n✅ Classes detected:", train_gen.class_indices)
print("\n🚀 Starting initial training...\n")

# === Train the top layers ===
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Fine-tune deeper layers ===
print("\n🔧 Fine-tuning the model...\n")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=3)

# === Save model and class mapping ===
os.makedirs("model", exist_ok=True)
model.save("model/food_quality_model.h5")

with open("model/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("\n✅ Model saved to model/food_quality_model.h5")
print("✅ Class indices saved to model/class_indices.json")
print("🎉 Training complete! You can now use app.py for predictions.")
