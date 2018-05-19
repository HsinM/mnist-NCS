
# coding: utf-8

# In[1]:


from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf


# In[2]:


model_file = "model.json"
weights_file = "weights.h5"


# In[3]:


with open(model_file, "r") as file:
    config = file.read()


# In[4]:


K.set_learning_phase(0)
model = model_from_json(config)
model.load_weights(weights_file)


# In[5]:


saver = tf.train.Saver()
sess = K.get_session()
saver.save(sess, "./TF_Model/tf_model")


# In[6]:


fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()

