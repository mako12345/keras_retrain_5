from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os,random
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import SGD
from keras.optimizers import Adam

import wx

batch_size=32
first = "C:/Users/andy/python_ws"

file_name='inceptionv3_gotoubun_fine'
label=['ichika','itsuki','miku','nino','yotsuba']
target_image = ""

app = wx.App()

# フォルダ選択ダイアログを作成
#dialog = wx.DirDialog(None, u'出力ファイルが入ったフォルダを選択してください', defaultPath=first)

#フォルダ選択ダイアログを表示
#dialog.ShowModal()

#dialog.GetPath()

# ファイル選択ダイアログを作成
dialog2 = wx.FileDialog(None, u'ファイルを選択してください', first)

# ファイル選択ダイアログを表示
dialog2.ShowModal()

target_image = dialog2.GetPath()

#load model and weights
json_string=open(file_name+'.json').read()
model=model_from_json(json_string)
model.load_weights(file_name+'.h5')

'''
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''

model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ["accuracy"]
)

temp_img=load_img(target_image,target_size=(224,224))
temp_img_array=img_to_array(temp_img)
temp_img_array=temp_img_array.astype('float32')/255.0
temp_img_array=temp_img_array.reshape((1,224,224,3))

img_pred=model.predict(temp_img_array)
print(label[np.argmax(img_pred)])
#print(img_pred)

for i in range(5):
    print(str(label[i]) + " " + str(img_pred[0][i]))