# Intel® Movidius™ NCS MNIST example

## Practice NCS using MNIST dataset with Keras

Train a simple CNN for MNIST using script

```
$ python train-mnist.py
```

Train a simple CNN for MNIST using jupyter

```
train-mnist.ipynb
```

Convert Keras model to Tensorflow model using script (model.json and weights.h5 file)

```
$ python convert-mnist.py
```

Convert Keras model to Tensorflow model using jupyter

```
convert-mnist.ipynb
```

Convert Keras model to Tensorflow model using script (model.h5 file)

```
$ python convert-mnist-only-h5.py
```

Convert Keras model to Tensorflow model using jupyter

```
convert-mnist-only-h5.ipynb
```

Check, Compile, Profile MNIST model using mvNC Toolkits

```
$ mvNCCheck TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
$ mvNCCompile TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
$ mvNCProfile TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
```

Do prediction on a random image using NCS

```
$ python predict-mnist-ncs.py
```



model.json

```
Only contain model graph (TF 1.8.0 format)
```

weights.h5

```
Only contain model weights
```

model.h5

```
Both contain model graph  (TF 1.8.0 format) and weights
```


## Reference

+ [oraoto/learn_ml](https://github.com/oraoto/learn_ml/blob/master/ncs/README.md)
+ [ardamavi/Intel-Movidius-NCS-Keras](https://github.com/ardamavi/Intel-Movidius-NCS-Keras)
