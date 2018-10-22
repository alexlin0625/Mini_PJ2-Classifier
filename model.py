import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from tensorflow.python.keras import Model, Sequential, optimizers

# To cross-check whether the instance is on GPU or not 
tf.test.gpu_device_name()

# The architecture of the CNN Model
model = Sequential()
model.add(BatchNormalization(input_shape=(224, 224, 3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Setting up some hyperparameters
batch_size = 128
epochs = 20
learning_rate = 0.001
num_classes = 120

# Training the model
opt = optimizers.Adam(lr = learning_rate)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
m = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                  batch_size=batch_size,verbose=2, epochs=epochs, shuffle = True)















