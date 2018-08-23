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
