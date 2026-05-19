import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("Semua library berhasil di-import.")
print("TensorFlow version:", tf.__version__)

from google.colab import drive
drive.mount('/content/drive')

dataset_path = "/content/drive/MyDrive/rockpaperscissors"

print(f"Path dataset: {dataset_path}")


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

print("generator training & validasi.")

class_names = list(train_generator.class_indices.keys())
print("Label kelas:", train_generator.class_indices)
print("   Urutan kelas:", class_names)

model = Sequential([
    # Blok 1: Ekstraksi fitur dasar (tepi, warna)
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    # Blok 2: Fitur lebih kompleks (tekstur, bentuk)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Blok 3: Fitur tingkat tinggi
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten: Ubah peta fitur 2D menjadi vektor 1D
    Flatten(),

    # Fully Connected (Dense) layer untuk klasifikasi
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
print("Model berhasil di-compile.")

print("\n=== Memulai pelatihan model ===\n")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)
print("Pelatihan selesai")

val_loss, val_acc = model.evaluate(validation_generator)
print(f"\nValidation Loss    : {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

predictions = model.predict(validation_generator)

print("Contoh probabilitas prediksi (5 sampel pertama):")
for i in range(5):
    pred_probs = predictions[i]
    pred_label = class_names[np.argmax(pred_probs)]
    prob_str = {class_names[j]: f"{pred_probs[j]:.4f}" for j in range(3)}
    print(f"  Sampel {i+1}: {prob_str} => Prediksi: {pred_label}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Akurasi
axes[0].plot(history.history['accuracy'], label='Train Acc', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Val Acc', marker='o')
axes[0].set_title('Akurasi per Epoch')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', marker='o', color='orange')
axes[1].plot(history.history['val_loss'], label='Val Loss', marker='o', color='red')
axes[1].set_title('Loss per Epoch')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("Grafik disimpan sebagai 'training_history.png'")

validation_generator.reset()
y_true = validation_generator.classes
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi'); plt.ylabel('Aktual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

validation_generator.reset()
images, labels = next(validation_generator)
batch_preds = model.predict(images)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(images[i])
    true_label = class_names[np.argmax(labels[i])]
    pred_label = class_names[np.argmax(batch_preds[i])]
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f"Aktual: {true_label}\nPrediksi: {pred_label}", color=color, fontsize=9)
    axes[i].axis('off')

plt.suptitle('Contoh Prediksi (Hijau = Benar, Merah = Salah)', fontsize=13)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.show()
print("Contoh prediksi disimpan sebagai 'sample_predictions.png'")

def predict_single_image(image_path):
    """
    Memprediksi kelas rock/paper/scissors dari satu file gambar.
    """
    # Muat gambar, resize ke 150x150
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)
    pred_label = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Tampilkan
    plt.figure(figsize=(4, 4))
    plt.imshow(load_img(image_path, target_size=(150, 150)))
    plt.title(f"Prediksi: {pred_label}\nKepercayaan: {confidence:.2f}%", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Hasil Prediksi: {pred_label} (Kepercayaan: {confidence:.2f}%)")
    return pred_label

