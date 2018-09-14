import numpy as np
from keras.models import Sequential
from keras.models import Model, load_model
from keras import backend as K
import struct

def save_weights(fName, model):
    with open(fName, 'wb') as fout:
        for layer in model.layers:
            d = layer.get_config()
            layerClass = layer.__class__.__name__
            if layerClass == 'Conv2D':
                convW, convH = d['kernel_size']
                filtersCount = d['filters']
                w = layer.get_weights()[0]
                b =  [0.0] * filtersCount
                if (d['use_bias']):                    
                    b = layer.get_weights()[1]                    
                fout.write(struct.pack("i", 1))
                fout.write(struct.pack("i", filtersCount))
                for i in range(0, filtersCount):
                    fout.write(struct.pack("f", b[i]))

                W, H, C, N = w.shape
                fout.write(struct.pack("i", 4))
                fout.write(struct.pack("i", W))
                fout.write(struct.pack("i", H))
                fout.write(struct.pack("i", C))
                fout.write(struct.pack("i", N))            
                for x in range(0, convW):                
                    for y in range(0, convH):                    
                        for cn in range(0, C):                        
                            for f in range(0, N):
                                fout.write(struct.pack("f", w[x, y, cn, f]))
                
            elif layerClass == 'Dense':
                w = layer.get_weights()
                srcNmb, dstNmb = w[0].shape
                fout.write(struct.pack("i", 1))
                fout.write(struct.pack("i", dstNmb))
                for y in range(0, dstNmb):
                    fout.write(struct.pack("f", w[1][y]))
                
                fout.write(struct.pack("i", 2))
                fout.write(struct.pack("i", srcNmb))
                fout.write(struct.pack("i", dstNmb))
                for x in range(0, srcNmb):                
                    for y in range(0, dstNmb):
                        fout.write(struct.pack("f", w[0][x, y]))

            elif layerClass == 'BatchNormalization':
                filtersCount = int(list(layer.output.shape)[-1])
                # order: gamma, betta, mean, std
                gamma = [1.0] * filtersCount
                betta = [0.0] * filtersCount
                shift = 0
                if d['scale']:
                    gamma = layer.get_weights()[0]
                    shift += 1
                if d['center']:
                    betta = layer.get_weights()[shift]
                    shift += 1
                mean = layer.get_weights()[shift]
                std = layer.get_weights()[shift + 1]
                for arr in [gamma, betta, mean, std]:
                    fout.write(struct.pack("i", 1))
                    fout.write(struct.pack("i", filtersCount))
                    for j in arr:
                        fout.write(struct.pack("f", j))

            elif layerClass =='DepthwiseConv2D':
                w = layer.get_weights()[0]
                W, H, C, N = w.shape
                b =  [0.0] * N
                if (d['use_bias']):
                    b = layer.get_weights()[1]
                    if len(b) != N:
                        print('Expected: %d real: %d' % (N, len(b)))
                fout.write(struct.pack("i", 1))
                fout.write(struct.pack("i", N))
                for i in range(0, N):
                    fout.write(struct.pack("f", b[i]))
                
                fout.write(struct.pack("i", 4))
                fout.write(struct.pack("i", W))
                fout.write(struct.pack("i", H))
                fout.write(struct.pack("i", C))
                fout.write(struct.pack("i", N))
                for x in range(0, W):
                    for y in range(0, H):
                        for cn in range(0, C):
                            for f in range(0, N):
                                fout.write(struct.pack("f", w[x, y, cn, f]))

            elif layerClass =='SeparableConv2D':
                for i in range(0, 2):
                    w = layer.get_weights()[i]
                    W, H, C, N = w.shape
                    
                    fout.write(struct.pack("i", 4))
                    fout.write(struct.pack("i", W))
                    fout.write(struct.pack("i", H))
                    fout.write(struct.pack("i", C))
                    fout.write(struct.pack("i", N))
                    for x in range(0, W):
                        for y in range(0, H):
                            for cn in range(0, C):
                                for f in range(0, N):
                                    fout.write(struct.pack("f", w[x, y, cn, f]))

                w = layer.get_weights()[1]
                W, H, C, N = w.shape

                b = [0.0] * N
                if (d['use_bias']):
                    b = layer.get_weights()[2]
                    if len(b) != N:
                        print('Expected: %d real: %d' % (N, len(b)))
                fout.write(struct.pack("i", 1))
                fout.write(struct.pack("i", N))
                for i in range(0, N):
                    fout.write(struct.pack("f", b[i]))

            elif layerClass == 'Conv2DTranspose':
                convW, convH = d['kernel_size']
                filtersCount = d['filters']
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
                fout.write(struct.pack("i", 1))
                fout.write(struct.pack("i", filtersCount))

                for i in range(0, filtersCount):
                    fout.write(struct.pack("f", b[i]))

                W, H, N, C = w.shape
                fout.write(struct.pack("i", 4))
                fout.write(struct.pack("i", W))
                fout.write(struct.pack("i", H))
                fout.write(struct.pack("i", C))
                fout.write(struct.pack("i", N))
                for x in range(0, convW):
                    for y in range(0, convH):
                        for cn in range(0, C):
                            for f in range(0, N):
                                fout.write(struct.pack("f", w[x, y, f, cn]))

            elif layerClass == 'PReLU':
                w = layer.get_weights()[0]
                if len(list(w.shape)) == 1:
                    xl = w.shape[0]
                    fout.write(struct.pack("i", 1))
                    fout.write(struct.pack("i", xl))
                    for x in range(0, xl):
                        fout.write(struct.pack("f", w[x]))
                elif len(list(w.shape)) == 3:
                    xl, yl, zl = w.shape
                    fout.write(struct.pack("i", 3))
                    fout.write(struct.pack("i", xl))
                    fout.write(struct.pack("i", yl))
                    fout.write(struct.pack("i", zl))
                    for x in range(0, xl):
                        for y in range(0, yl):
                            for z in range(0, zl):
                                fout.write(struct.pack("f", w[x, y, z]))
                else:
                    print("PRELU unexpected len ")
                    print(w.shape)
                

            # Layers, that have no weights:
            elif layerClass == 'MaxPooling2D' or layerClass == 'Flatten' or layerClass == 'Dropout':
                continue
            elif layerClass == 'Concatenate' or layerClass == 'InputLayer' or layerClass == 'Activation':
                continue
            elif layerClass == 'ZeroPadding2D' or layerClass == 'DepthwiseConv2D' or layerClass == 'AveragePooling2D':
                continue
            elif layerClass == 'Reshape' or layerClass == 'GlobalAveragePooling2D' or layerClass == 'Add':
                continue

            else:
                print("UNKNOWN WEIGHT: %s %s\n" % (d['name'], layerClass))

def get_prev_layers(layer):
    arr = []
    for x in layer._inbound_nodes:
        for j in x.inbound_layers:
            arr.append(j)
    return arr

def csharp_activation(node_name, activation_name, dim):
    if activation_name == 'linear':
        return ""
    elif activation_name == 'softmax':
        return "    SoftMax%sD(%s);" % (dim, node_name)
    elif activation_name == 'relu':
        return "    ReLu%sD(%s);" % (dim, node_name)
    elif activation_name == 'relu6':
        return "    ReLu6_%sD(%s);" % (dim, node_name)    
    elif activation_name == 'sigmoid':
        return "    Sigmoid%sD(%s);" % (dim, node_name)
    else:
        print ("UNKNOWN ACTIVATION %s !!!" % (activation_name))
        return "UNKNOWN ACTIVATION %s !!!" % (activation_name)

def sequential_to_csharp(model, csClassName, debug_console_write = False):
    script_loader = []
    script_loader.append("protected void Load_Weights(string fName)")
    script_loader.append("{")
    script_loader.append("    using (FileStream fs = new FileStream(fName, FileMode.Open, FileAccess.Read))")
    script_loader.append("    using(BinaryReader br = new BinaryReader(fs))")
    script_loader.append("    {")
    
    script = []
    
    model_output_type = 'float[%s]' % (',' * (len(np.array(model.output).shape)-2))    
    layer_nmb = 0
    prev_name = ''
    lname = ''
    for layer in model.layers:        
        layer_nmb = layer_nmb + 1
        d = layer.get_config()
        lname = d['name'].replace('-', '_')

        output_type = 'float[%s]' % (',' * (len(layer.output.shape) - 2))
        output_dim = len(layer.output.shape) - 1

        # Try to found previous layers
        arr = []        
        for l in get_prev_layers(layer):
            t = l
            if l.__class__.__name__ == 'Dropout':
                t = get_prev_layers(l)[0]            
            arr.append(t.get_config()['name'].replace('-', '_'))
        if (len(arr) > 0):
            prev_name = ", ".join(arr)        

        if layer_nmb==1: # If first layer in model
            commas = len(d['batch_input_shape']) - 2 # comma counter            
            script.append("public %s Process(float[%s] src)" % (model_output_type, "," * commas));
            script.append("{");
            prev_name = 'src'

        layerClass = layer.__class__.__name__
        if layerClass == 'Conv2D':
            #print(d)
            strideX, strideY = d['strides']
            if (d['padding'] == 'same'):
                script.append("    %s %s = Conv2d((float[,,,])Weights[\"%s_weights\"], (float[])Weights[\"%s_bias\"], %s, true, %s, %s);" % (output_type, lname, lname, lname, prev_name, strideX, strideY))
            else:
                script.append("    %s %s = Conv2d((float[,,,])Weights[\"%s_weights\"], (float[])Weights[\"%s_bias\"], %s, false, %s, %s);" % (output_type, lname, lname, lname, prev_name, strideX, strideY))
            script_loader.append("        Weights.Add(\"%s_bias\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_weights\", ReadTensor(br));" % (lname))            
            s = csharp_activation(lname, d['activation'], 3)
            if (len(s) > 0) :
                script.append(s)

        elif layerClass == 'PReLU':
            script.append("    %s %s = PReLU%dD((%s)Weights[\"%s_weights\"], %s);" % (output_type, lname, output_dim, output_type, lname, prev_name))
            script_loader.append("        Weights.Add(\"%s_weights\", ReadTensor(br));" % (lname))

        elif layerClass == 'ZeroPadding2D':
            hp, wp = d['padding']
            top, bottom = hp
            left, right = wp
            script.append("    %s %s = ZeroPadding2D(%s, %s, %s, %s, %s);" % (output_type, lname, prev_name, top, bottom, left, right))

        elif layerClass == 'Reshape':
            l = list(layer.output.shape)[1:]
            arr = []            
            for i in l:
                arr.append(str(i))
            script.append("    %s %s = (%s)Reshape(%s, %s);" % (output_type, lname, output_type, prev_name, ','.join(arr)))

        elif layerClass == 'Permute':
            dims = ','.join([str(i) for i in list(d['dims'])])            
            script.append("    %s %s = Permute%dD(%s, %s);" % (output_type, lname, output_dim, prev_name, dims))

        elif layerClass == 'BatchNormalization':
            script_loader.append("        Weights.Add(\"%s_gamma\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_betta\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_mean\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_std\", ReadTensor(br));" % (lname))
            script.append("    %s %s = BatchNormalization%sD(%s, (float[])Weights[\"%s_gamma\"], (float[])Weights[\"%s_betta\"], (float[])Weights[\"%s_mean\"], (float[])Weights[\"%s_std\"], %sF, %sF);"
                          % (output_type, lname, output_dim, prev_name, lname, lname, lname, lname, d["epsilon"], d["momentum"]))

        elif layerClass == 'Activation':
            script.append("    %s %s = (%s)%s.Clone();" % (output_type, lname, output_type, prev_name))
            script.append(csharp_activation(lname, d['activation'], output_dim))

        elif layerClass == 'InputLayer':
            script.append("    %s %s = src;" % (output_type, lname))
            
        elif layerClass == 'Concatenate':
            script.append("    %s %s = Concatenate%sD(%s);" % (output_type, lname, output_dim, prev_name))
            
        elif layerClass == 'Add':
            script.append("    %s %s = Add%sD(%s);" % (output_type, lname, output_dim, prev_name))            

        elif layerClass == 'AveragePooling2D':
            s1, s2 = d['strides']
            p1, p2 = d['pool_size']
            if (d['padding'] == 'same'):
                script.append("    %s %s = AveragePooling2D(%s, %s, %s, %s, %s, true);" % (output_type, lname, p1, p2, prev_name, s1, s2))
            else:
                script.append("    %s %s = AveragePooling2D(%s, %s, %s, %s, %s, false);" % (output_type, lname, p1, p2, prev_name, s1, s2))
            
        elif layerClass == 'MaxPooling2D':
            #print(d)
            s1, s2 = d['strides']
            p1, p2 = d['pool_size']
            if (d['padding'] == 'same'):
                script.append("    %s %s = MaxPool2d(%s, %s, %s, %s, %s, true);" % (output_type, lname, p1, p2, s1, s2, prev_name))
            else:
                script.append("    %s %s = MaxPool2d(%s, %s, %s, %s, %s, false);" % (output_type, lname, p1, p2, s1, s2, prev_name))

        elif layerClass == 'Dense':
            script.append("    %s %s = Dense1D(%s, (float[,])Weights[\"%s_weights\"], (float[])Weights[\"%s_bias\"]);" % (output_type, lname, prev_name, lname, lname))
            script_loader.append("        Weights.Add(\"%s_bias\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_weights\", ReadTensor(br));" % (lname))
            s = csharp_activation(lname, d['activation'], 1)
            if (len(s) > 0) :
                script.append(s)

        elif layerClass =='DepthwiseConv2D':
            strideX, strideY = d['strides']
            if (d['padding'] == 'same'):
                script.append("    %s %s = DepthwiseConv2D((float[,,,])Weights[\"%s_weights\"], (float[])Weights[\"%s_bias\"], %s, true, %s, %s);" % (output_type, lname, lname, lname, prev_name, strideX, strideY))
            else:
                script.append("    %s %s = DepthwiseConv2D((float[,,,])Weights[\"%s_weights\"], (float[])Weights[\"%s_bias\"], %s, false, %s, %s);" % (output_type, lname, lname, lname, prev_name, strideX, strideY))
            script_loader.append("        Weights.Add(\"%s_bias\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_weights\", ReadTensor(br));" % (lname))            
            s = csharp_activation(lname, d['activation'], 3)
            if (len(s) > 0) :
                script.append(s)

        elif layerClass =='SeparableConv2D':
            strideX, strideY = d['strides']
            if (d['padding'] == 'same'):
                script.append("    %s %s = SeparableConv2D((float[,,,])Weights[\"%s_weights1\"], (float[,,,])Weights[\"%s_weights2\"], (float[])Weights[\"%s_bias\"], %s, true, %s, %s);" % (output_type, lname, lname, lname, lname, prev_name, strideX, strideY))
            else:
                script.append("    %s %s = DepthwiseConv2D((float[,,,])Weights[\"%s_weights1\"], (float[,,,])Weights[\"%s_weights2\"], (float[])Weights[\"%s_bias\"], %s, false, %s, %s);" % (output_type, lname, lname, lname, lname, prev_name, strideX, strideY))
            script_loader.append("        Weights.Add(\"%s_weights1\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_weights2\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_bias\", ReadTensor(br));" % (lname))

            s = csharp_activation(lname, d['activation'], 3)
            if (len(s) > 0) :
                script.append(s)

        elif layerClass == 'Flatten':
            script.append("    %s %s = Flatten(%s);" % (output_type, lname, prev_name))

        elif layerClass == 'GlobalAveragePooling2D':
            script.append("    %s %s = GlobalAveragePooling2D(%s);" % (output_type, lname, prev_name))            

        elif layerClass == 'Conv2DTranspose':
            sx, sy = d['strides']
            script.append("    %s %s = Conv2DTr((float[,,,])Weights[\"%s_weights\"], (float[])Weights[\"%s_bias\"], %s, %s, %s);" % (output_type, lname, lname, lname, sx, sy, prev_name))
            script_loader.append("        Weights.Add(\"%s_bias\", ReadTensor(br));" % (lname))
            script_loader.append("        Weights.Add(\"%s_weights\", ReadTensor(br));" % (lname))
            s = csharp_activation(lname, d['activation'], 3)
            if (len(s) > 0) :
                script.append(s)
        elif layerClass == 'Dropout':
            continue
        else:            
            print("UNKNOWN: %s %s\n" % (d['name'], layerClass))

        if (debug_console_write):
            if (layer_nmb > 2):
                if output_dim == 3:
                    script.append("    System.Console.WriteLine($\"{NetUtils.CompareTensor3D(NetUtils.ReadTensor3D(@\"c:\\LAB\\DATA\\OCR\\WEIGHTS\\%s.dat\"), %s, 0.01F)} - %s \");" % (lname, lname, lname))
        
    script_loader.append('    }')
    script_loader.append('}')

    script.append("    return %s;" % (lname))
    script.append("}")

    result_script = []

    result_script.append('using System.IO;')
    result_script.append('namespace MyModel')
    result_script.append('{')
    result_script.append('    public class %s : NetBase' % (csClassName))
    result_script.append('    {')
    result_script.append('        public %s(string fName)' % (csClassName))
    result_script.append('        {')
    result_script.append('            Load_Weights(fName);')
    result_script.append('        }')
    script.extend(script_loader)
    for l in script:
        result_script.append('        %s' % (l))

    result_script.append('    }')
    result_script.append('}')
    return result_script

