#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

url = "http://127.0.0.1:5001/predict"
data = {"features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]}

response = requests.post(url, json=data)
print(response.json())  # Output: {'prediction': 1}


# In[ ]:




