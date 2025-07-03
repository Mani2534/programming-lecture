-------------Tensor flow Implementation------------

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

print("Initializing TensorFlow model training...")

# Step 1: Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

batch_sz = 32
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(batch_sz)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_sz)

# Step 2: Define the model using Sequential API
tf_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Step 3: Set up optimizer, loss, and accuracy metrics
loss_fn = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam()
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

@tf.function
def training_step(x, y):
    with tf.GradientTape() as tape:
        output = tf_model(x, training=True)
        loss = loss_fn(y, output)
    gradients = tape.gradient(loss, tf_model.trainable_variables)
    opt.apply_gradients(zip(gradients, tf_model.trainable_variables))
    accuracy_metric.update_state(y, output)
    return loss

# Step 4: Model training
num_epochs = 5
start_time = time.time()

for ep in range(num_epochs):
    print(f"\nTraining Epoch {ep + 1}/{num_epochs}")
    for idx, (batch_x, batch_y) in enumerate(train_ds):
        current_loss = training_step(batch_x, batch_y)
        if idx % 100 == 0:
            print(f"Batch {idx}, Loss: {current_loss.numpy():.4f}, Acc: {accuracy_metric.result().numpy():.4f}")
    print(f"Epoch {ep+1} Accuracy: {accuracy_metric.result().numpy():.4f}")
    accuracy_metric.reset_state()

total_time = time.time() - start_time
print(f"\nTotal training time (TensorFlow): {total_time:.2f} seconds")

# Step 5: Evaluation and Inference
tf_model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

print("\nRunning evaluation...")
eval_start = time.time()
eval_loss, eval_acc = tf_model.evaluate(test_ds, verbose=2)
eval_end = time.time()

print(f"TF Test Accuracy: {eval_acc:.4f}")
print(f"Inference Duration: {eval_end - eval_start:.2f} seconds")

# Step 6: Save as TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
lite_model = converter.convert()
with open("converted_model_tf.tflite", "wb") as model_file:
    model_file.write(lite_model)

-------------Pytorch Implementation-------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import time

print("\nInitializing PyTorch model training...")

# Step 1: Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Prepare the data
transform_pipeline = transforms.Compose([transforms.ToTensor()])
train_set = MNIST(root="./data", train=True, transform=transform_pipeline, download=True)
test_set = MNIST(root="./data", train=False, transform=transform_pipeline, download=True)

batch_sz = 32
train_loader = DataLoader(train_set, batch_size=batch_sz, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_sz, shuffle=False)

# Step 3: Define model
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

pt_model = FeedForwardNet().to(device)

# Step 4: Training setup
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(pt_model.parameters())

# Step 5: Training loop
epochs = 5
start_train = time.time()

for epoch in range(epochs):
    pt_model.train()
    total_correct = 0
    total_count = 0
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        preds = pt_model(images)
        loss = loss_func(preds, targets)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(preds.data, 1)
        total_correct += (predicted == targets).sum().item()
        total_count += targets.size(0)

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {total_correct / total_count:.4f}")

    epoch_acc = total_correct / total_count
    print(f"Epoch {epoch+1} Accuracy: {epoch_acc:.4f}")

train_duration = time.time() - start_train
print(f"\nTotal training time (PyTorch): {train_duration:.2f} seconds")

# Step 6: Evaluation
pt_model.eval()
correct = 0
total = 0
start_eval = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = pt_model(images)
        _, pred_classes = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred_classes == labels).sum().item()

end_eval = time.time()
accuracy = correct / total

print(f"PyTorch Test Accuracy: {accuracy:.4f}")
print(f"Inference Time: {end_eval - start_eval:.2f} seconds")

# Step 7: Save model state
torch.save(pt_model.state_dict(), "converted_model_pytorch.pth")
