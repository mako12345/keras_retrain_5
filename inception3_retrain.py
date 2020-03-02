from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import CSVLogger

n_categories=5
batch_size=32
train_dir='gotoubun/train'
validation_dir='gotoubun/validation'
file_name='inceptionv3_gotoubun_fine'

base_model = InceptionV3(
    include_top = False,
    weights = "imagenet",
    input_shape = None
)

# 全結合層の新規構築
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(n_categories, activation = 'softmax')(x)

# ネットワーク定義
model = Model(inputs = base_model.input, outputs = predictions)
print("{}層".format(len(model.layers)))


# 249層までfreeze
for layer in model.layers[:249]:
    layer.trainable = False

    # Batch Normalization の freeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True

#250層以降、学習させる
for layer in model.layers[249:]:
    layer.trainable = True

# layer.trainableの設定後にcompile
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ["accuracy"]
)

model.summary()

#save model
json_string=model.to_json()
open(file_name+'.json','w').write(json_string)

train_datagen=ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

hist=model.fit_generator(train_generator,
                         epochs=100,
                         verbose=1,
                         validation_data=validation_generator,
                         callbacks=[CSVLogger(file_name+'.csv')])

#save weights
model.save(file_name+'.h5')
