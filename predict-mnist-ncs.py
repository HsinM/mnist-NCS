from mvnc import mvncapi as mvnc
import numpy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Prepare test image
test_idx = numpy.random.randint(0, 10000)
test_image = x_test[test_idx]
test_image = test_image.astype('float32') / 255.0

# Using NCS Predict
devices = mvnc.EnumerateDevices()
device = mvnc.Device(devices[0])
device.OpenDevice()

with open("graph", mode='rb') as f:
    graphfile = f.read()

graph = device.AllocateGraph(graphfile)

graph.LoadTensor(test_image.astype('float16'), 'user object')

output, userobj = graph.GetResult()

graph.DeallocateGraph()
device.CloseDevice()

print("NCS", output, output.argmax())
print("Correct", y_test[test_idx])