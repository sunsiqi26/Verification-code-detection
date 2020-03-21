#Verification code detection

There are two main ways to detect the verification code:

- [1.Split](#env)
- [2.End-to-End](#env1)

This project has practiced and compared the two methods, the file structure:
```
├── Screen
│   ├── split method flowchart.png
│   ├── end-to-end method flowchart.jpg
│   └── acc+recall.png
├── code
│   ├── cut
│   │   ├── get_checkcode.py
│   │   ├── pre.py
│   │   ├── process.py
│   │   └── train.ipynb
│   └── endtoend
│       └── cnn_veri-1.py
└── readme.md
```

#<span id="env">1.Split</span>

### Dataset

1. Web crawling: https://www.cndns.com/common/GenerateCheckCode.aspx
2. Tags: OCR pre-classification, using the pytesseract library, unrecognized manual classification, one folder per character (the folder is named a character)

### Feature extract

Because the verification code contains numbers and letters, and the thickness and shape of the same character are different, different feature extraction methods are specifically considered here:

According to the matrix formed by the pixels, three features are extracted in each row, which are the number of consecutive occurrences of the nth 0 (pixels are black dots).

### Training

Using SVM for training, because the character denoising is not perfect, and the shape of the character changes, the accuracy rate is only 0.89 for the time being, we will consider a better denoising method afterwards.


### Test

The model can output test pictures correctly, see train.ipynb for details

#<span id="env1">2.End-to-End</span>

### Configuration environment

- captcha 0.3
- tensorflow 1.13.1
- numpy 
- tqdm 

### Dataset

We use captcha, a library that generates verification code that comes with Python, supports image verification code and voice verification code. Set the captcha format to numeric capital letters.


### Generator

Using Keras Sequence class to implement the data generator
```
class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=128, height=64):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)
    
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y
```
X shape (batch_size, height, width, 3)
y shape (batch_size, n_class)

### Build CNN

```
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
    for j in range(n_cnn):
        x = Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
model = Model(inputs=input_tensor, outputs=x)
```

The feature extraction part uses two convolutions, a pooled structure, repeating five blocks, and then flattening it, connecting four classifiers, each classifier is 36 neurons, and the probability of outputting 36 characters.

### Training

Use model.fit_generator, use the same generator to generate the validation set, use Adam optimizer, and set the learning rate to 1e-3.

The EarlyStopping method is used to automatically terminate the training after the loss does not drop more than 3 epochs.
Use the ModelCheckpoint method to save the best model during training.

```
callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv',append=True),ModelCheckpoint('cnn_best.h5', save_best_only=True)]
model.compile(loss='categorical_crossentropy',optimizer=Adam(1e-4, amsgrad=True),metrics=['accuracy'])
model.fit_generator(train_data, epochs=20,validation_data=valid_data, workers=4,use_multiprocessing=True,callbacks=callbacks)
```

### Result

<img style="width:250px;height:150px" src="https://pic.superbed.cn/item/5ddfc1f38e0e2e3ee9ee28a4.jpg"/>

```
Epoch 20/20
20/20 [==============================] - 54s 3s/step - loss: 1.0480 - c1_loss: 0.1663 - c2_loss: 0.3222 - c3_loss: 0.3558 - c4_loss: 0.2037 - c1_acc: 0.9512 - c2_acc: 0.9074 - c3_acc: 0.8914 - c4_acc: 0.9371
20/20 [==============================] - 275s 14s/step - loss: 0.8244 - c1_loss: 0.1481 - c2_loss: 0.2636 - c3_loss: 0.2652 - c4_loss: 0.1476 - c1_acc: 0.9500 - c2_acc: 0.9187 - c3_acc: 0.9285 - c4_acc: 0.9484 - val_loss: 1.0480 - val_c1_loss: 0.1663 - val_c2_loss: 0.3222 - val_c3_loss: 0.3558 - val_c4_loss: 0.2037 - val_c1_acc: 0.9512 - val_c2_acc: 0.9074 - val_c3_acc: 0.8914 - val_c4_acc: 0.9371
```
