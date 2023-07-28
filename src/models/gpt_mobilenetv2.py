import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense


class CustomMobileNetV2(tf.keras.Model):
    def __init__(self, num_classes=1000, alpha=1.0):
        super(CustomMobileNetV2, self).__init__()

        self.alpha = alpha

        # Define the MobileNetV2 architecture
        self.conv1 = Conv2D(filters=int(32 * self.alpha), kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        self.block1 = self._inverted_residual_block(int(16 * self.alpha), 1, 1)
        self.block2 = self._inverted_residual_block(int(24 * self.alpha), 2, 6)
        self.block3 = self._inverted_residual_block(int(32 * self.alpha), 3, 6)
        self.block4 = self._inverted_residual_block(int(64 * self.alpha), 4, 6)
        self.block5 = self._inverted_residual_block(int(96 * self.alpha), 3, 6)
        self.block6 = self._inverted_residual_block(int(160 * self.alpha), 3, 6)
        self.block7 = self._inverted_residual_block(int(320 * self.alpha), 1, 6)

        self.conv2 = Conv2D(filters=int(1280 * self.alpha), kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation='softmax')

    def _inverted_residual_block(self, filters, expansion, strides):
        out_channels = int(filters * self.alpha)

        return tf.keras.Sequential([
            Conv2D(filters=expansion * out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormalization(),
            ReLU(),
            DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same'),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormalization()
        ])

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


# You can specify the number of classes and alpha (width multiplier) as desired
num_classes = 1000  # Adjust this based on your classification problem
alpha = 1.0  # Adjust this to control the width of the network

model = CustomMobileNetV2(num_classes=num_classes, alpha=alpha)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with your data (X_train and y_train are placeholders, use your own data)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
