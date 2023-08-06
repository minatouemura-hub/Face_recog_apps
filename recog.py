import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import numpy as np
import time

classes = ["cool","cute","femini","fresh"]
nb_classes = len(classes)
train_data_dir="photo/train"
val_data_dir = "photo/val"
img_height , img_width=224,224
target_size = (224,224)
image_files = os.listdir(train_data_dir)


batch_size = 32
nb_epoch = 50

train_datagen = ImageDataGenerator(
    rescale= 1.0/255,
    zoom_range= 0.2,
    horizontal_flip= True
)
validation_datagen = ImageDataGenerator(
    rescale=1.0/255
    )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = target_size,
    color_mode='rgb',
    classes=classes,
    class_mode= 'categorical',
    batch_size=batch_size,
    shuffle=True)

val_genarator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size = target_size,
    color_mode='rgb',
    classes=classes,
    class_mode= 'categorical',
    batch_size=batch_size,
    shuffle=True
)
#keras を用いてモデル構築
input_tensor = Input(shape=(img_width,img_height,3))
vgg16 = VGG16(include_top=False,weights='imagenet',input_tensor=input_tensor)
#include_top　= false でもとの1000クラス分類を利用するかを決定する

top_model = Sequential()
top_model.add(Flatten(input_shape =vgg16.output_shape[1:]))
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes,activation='softmax'))

vgg_model = Model(vgg16.input, top_model(vgg16.output))

# VGG16の図の青色の部分は重みを固定（frozen）
for layer in vgg_model.layers[:15]:
    layer.trainable = False

# 多クラス分類を指定
vgg_model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.legacy.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])

steps_per_epoch = train_generator.n
validation_steps = val_genarator.n

history = vgg_model.fit(
                        train_generator,
                        steps_per_epoch= steps_per_epoch // batch_size,#何周すれば受け取ったデータの全てを見たことになるのか
                        epochs = nb_epoch,#何周分学習するのか
                        validation_data=val_genarator,
                        validation_steps=validation_steps//batch_size,
                        verbose=1,#学習の進捗を表示してくれる
                        )

vgg_model.save("bottleneck_fc_model.h5")

