import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import requests

dataset_folder = '/content/PEANUTFINAL'
train_images, train_labels, test_images, test_labels = [], [], [], []

def process_images(image_folder, images, labels):
    for folder_name in os.listdir(os.path.join(dataset_folder, image_folder)):
        folder_path = os.path.join(dataset_folder, image_folder, folder_name)
        for filename in os.listdir(folder_path):
            image = cv2.imread(os.path.join(folder_path, filename))
            image = cv2.resize(image, (32, 32)) / 255.0
            images.append(image)
            labels.append(int(folder_name[1]))
process_images('training', train_images, train_labels)
process_images('testing', test_images, test_labels)

train_images, train_labels, test_images, test_labels = np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)
class_names = ['g1', 'g2', 'g3', 'g4']

plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])  # Assuming train_images is already loaded
    label_index = train_labels[i] - 1  # Adjusting label index to match class_names indices
    if label_index >= 0 and label_index < len(class_names):
        plt.xlabel(class_names[label_index])
plt.show()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 15, 15, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 6, 6, 64)          0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
 flatten (Flatten)           (None, 1024)              0         
                                                                 
 dense (Dense)               (None, 64)                65600     
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
 flatten_1 (Flatten)         (None, 10)                0         
                                                                 
 dense_2 (Dense)             (None, 64)                704       
                                                                 
 dense_3 (Dense)             (None, 10)                650       
                                                                 
=================================================================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Adjust batch size to match labels
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, # Add batch_size here
                    validation_data=(test_images, test_labels))
model.save("peanut_model.h5")
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,
                                     test_labels,
                                     verbose=2)

print('Test Accuracy is',test_acc)
Test Accuracy is 0.96875

# Load your trained model
model = tf.keras.models.load_model('peanut_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('peanut_model.tflite', 'wb') as f:
    f.write(tflite_model)



# Load your trained model
model = tf.keras.models.load_model('/content/peanut_model.h5')
def preprocess_image(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    return img
def send_to_thingspeak(predicted_index):
    channel_id = 2584958
    api_key = 'api_key'
    url = f'https://api.thingspeak.com/update?api_key={api_key}'
    url += f'&field1={predicted_index}'
    response = requests.get(url)
    if response.ok:
        print('Result posted to ThingSpeak successfully.')
    else:
        print('Failed to post result to ThingSpeak.')
output_path = '/content/Images'
os.makedirs(output_path, exist_ok=True)

num_images = int(input("Enter the number of images to capture: "))
interval = 10  # Interval between captures (in seconds)
for i in range(num_images):
    js = Javascript('''
    async function takePhoto(quality) {
      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(video);
      video.srcObject = stream;
      await video.play();

      await new Promise(resolve => setTimeout(resolve, 2000));  // Wait for 2 seconds

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      video.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(0.8))
    binary = b64decode(data.split(',')[1])
    filename = f'image_{time.strftime("%Y%m%d%H%M%S")}.jpg'
    filename = os.path.join(output_path, filename)
    with open(filename, 'wb') as f:
        f.write(binary)

    test_image = preprocess_image(binary)
    predictions = model.predict(tf.expand_dims(test_image, axis=0))
    predicted_class_index = tf.argmax(predictions, axis=1)[0].numpy()
    print(f"Image captured and predicted class index: {predicted_class_index}")
    send_to_thingspeak(predicted_class_index)
    if i < num_images - 1:
        time.sleep(interval)
