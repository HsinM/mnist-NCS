
# coding: utf-8

# In[1]:


from keras.models import load_model
from keras import backend as K
import tensorflow as tf


# In[2]:


model_weights_file = "model.h5"


# In[3]:


K.set_learning_phase(0)
model = load_model(model_weights_file)


# In[4]:


saver = tf.train.Saver()
sess = K.get_session()
saver.save(sess, "./TF_Model/tf_model")


# In[5]:


# For Tensorboard
#fw = tf.summary.FileWriter('logs', sess.graph)
#fw.close()

