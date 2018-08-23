# keras-exporter
This tool can export Keras model to C#

# Quick start to export Keras model to C#:

1. Export Weights and CS code like this:
```python
import struct
import keras2cs

from keras.applications mobilenet

folderName = "GeneratedCode\\"

model = mobilenet.MobileNet(weights='imagenet') # load model from keras
model.summary() # print model summary
keras2cs.save_weights('%sMobileNet.dat' % (folderName), model) # save MobileNet.dat weights file
main_script = keras2cs.model_to_csharp(model, "MobileNet") # get C# lines for MobileNet.cs file
with open('%sMobileNet.cs' % (folderName), 'w') as fout:
    fout.write("\n".join(main_script))
```
2. Copy NetBase.cs and NetUtils.cs file near to generated class and weights class

3. Use generated class like this:

```C#
var net = new MobileNet("MobileNet.dat");
float[,,] img = NetUtils.PrepareImageMobileNet("test_dog.png");
float[] prediction = net.Process(img);
Console.WriteLine("Top 3 results: " + string.Join(", ", NetUtils.DecodeImageNetResult(prediction, 3)));
```
output will be like this:
```
Top 3 results: pug [0,9969646], Brabancon_griffon [0,002718836], French_bulldog [0,0001286689]
```
# Supported layers:
* Reshape
* GlobalAveragePooling2D
* Flatten
* Concatenate3D
* Add
* BatchNormalization3D
* ZeroPadding2D 
* Conv2d
* DepthwiseConv2D
* SeparableConv2D
* AveragePooling2D
* MaxPool2d
* Dense
* Conv2DTr

# Example.bat
This exmaple do:
1. Create GeneratedCode folder
2. Run generate_example.py to create nets: ResNet50, InceptionV3, Xception, MobileNet
3. Copy:
- Program.cs - file, that test all of this networks on test_dog image
- NetBase.cs - file, that contains all layers implimentations
- NetUtils.cs - file for image preprocessing and result decode
- project.csproj - general progect file to build test application
4. Find MSBuild to build test app
5. Try to build test app
6. Run test app (it will measure executing time of all network and print top-3 predictions)

output like this:
```
ResNet50...
Time: 14,854 s
Top 3 results: pug [0,973725], Brabancon_griffon [0,01610618], Pekinese [0,00580
9539]
--------------

InceptionV3...
Time: 17,658 s
Top 3 results: pug [0,4592525], Brabancon_griffon [0,1961291], Pekinese [0,07487
778]
--------------

MobileNet...
Time: 2,370 s
Top 3 results: pug [0,9969646], Brabancon_griffon [0,002718836], French_bulldog
[0,0001286689]
--------------

Xception...
Time: 30,489 s
Top 3 results: pug [0,8761638], Brabancon_griffon [0,02947094], Pekinese [0,0016
53932]
--------------

Press a key
```
