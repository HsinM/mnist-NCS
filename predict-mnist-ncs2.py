#!/usr/bin/env python3

#https://movidius.github.io/ncsdk/ncapi/python_api_migration.html

import mvnc.mvncapi as fx
import mnist    # pip3 install mnist
import numpy

# For tensorflow
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()


# Prepare test image
test_idx = numpy.random.randint(0, 10000)
test_image = x_test[test_idx]
test_image = test_image.astype('float32') / 255.0

# Using NCS Predict
# set the logging level for the NC API
fx.global_set_option(fx.GlobalOption.RW_LOG_LEVEL, 0)

# get a list of names for all the devices plugged into the system
devices = fx.enumerate_devices()
if (len(devices) < 1):
    print("Error - no NCS devices detected, verify an NCS device is connected.")
    quit()

# get the first NCS device by its name.  For this program we will always open the first NCS device.
dev = fx.Device(devices[0])

# try to open the device.  this will throw an exception if someone else has it open already
try:
    dev.open()
except:
    print("Error - Could not open NCS device.")
    quit()

# Read a compiled network graph from file (set the graph_filepath correctly for your graph file)
with open("graph", mode='rb') as f:
    graphfile = f.read()

graph = fx.Graph(graph1)

# Allocate the graph on the device and create input and output Fifos
in_fifo, out_fifo = graph.allocate_with_fifos(dev, graphfile)

# Write the input to the input_fifo buffer and queue an inference in one call
graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, test_image, 'user object')

# Read the result to the output Fifo
output, userobj = out_fifo.read_elem()

# Deallocate and destroy the fifo and graph handles, close the device, and destroy the device handle
try:
    in_fifo.destroy()
    out_fifo.destroy()
    graph.destroy()
    dev.close()
    dev.destroy()
except:
    print("Error - could not close/destroy Graph/NCS device.")
    quit()

print("NCS", output, output.argmax())
print("Correct", y_test[test_idx])
