
import numpy as np
from keras.models import Sequential
from keras.models import Model, load_model
from keras import backend as K
import tensorflow as tf
from keras import backend as K
import struct
import keras2cs

from keras.applications import vgg16, inception_v3, resnet50, mobilenet, xception

folderName = "GeneratedCode\\"

model = resnet50.ResNet50(weights='imagenet')
model.summary()
keras2cs.save_weights('%sResNet50.dat' % (folderName), model)
main_script = keras2cs.sequential_to_csharp(model, "ResNet50")
with open('%sResNet50.cs' % (folderName), 'w') as fout:
    fout.write("\n".join(main_script))

model = inception_v3.InceptionV3(weights='imagenet')
model.summary()
keras2cs.save_weights('%sInceptionV3.dat' % (folderName), model)
main_script = keras2cs.sequential_to_csharp(model, "InceptionV3")
with open('%sInceptionV3.cs' % (folderName), 'w') as fout:
    fout.write("\n".join(main_script))

model = xception.Xception(weights='imagenet')
model.summary()
keras2cs.save_weights('%sXception.dat' % (folderName), model)
main_script = keras2cs.sequential_to_csharp(model, "Xception", False)
with open('%sXception.cs' % (folderName), 'w') as fout:
    fout.write("\n".join(main_script))

model = mobilenet.MobileNet(weights='imagenet')
model.summary()
keras2cs.save_weights('%sMobileNet.dat' % (folderName), model)
main_script = keras2cs.sequential_to_csharp(model, "MobileNet")
with open('%sMobileNet.cs' % (folderName), 'w') as fout:
    fout.write("\n".join(main_script))
    

