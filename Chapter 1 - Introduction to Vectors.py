#!/usr/bin/env python
# coding: utf-8

# # Chapter 1: Introduction to Vectors
# 

# The heart of linear algebra lies in two operations:
# 1. Sum
# 2. Product
# 
# Let there be two vectors $\mathbf{x}$ and $\mathbf{y}$
# 
# We can add them:
#  $\mathbf{x} + \mathbf{y}$
# 
# Let there be two constants $c$ and $d$<br> 
# We can multiply each vector by each constant to get: $c\mathbf{x} $ and $d\mathbf{y}$<br>
# A linear combination is the combination of the two operations: $c\mathbf{x} + d\mathbf{y}$

# In[14]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go


# In[12]:


## vectors
v = np.array([1,1])
w = np.array([2,3])

## coefficients
c = 1
d = 1

linear_combination = c*v + d*w
print("Linear combination cv + dw = ", linear_combination)


# In[33]:


# Create a figure
fig = go.Figure()

# Add vector v
fig.add_trace(go.Scatter(x=[0 ,v[0]], y=[0,v[1]],
                         mode = 'lines+markers', name = 'v',
                         line = dict(color = 'blue', width = 3),
                         marker = dict(size = 8)))

# Add vector w
fig.add_trace(go.Scatter(x=[0 ,w[0]], y=[0,w[1]],
                         mode = 'lines+markers', name = 'w',
                         line = dict(color = 'orange', width = 3),
                         marker = dict(size = 8)))


# Add the sum vector (v + w)
fig.add_trace(go.Scatter(x=[0 ,linear_combination[0]], y=[0,linear_combination[1]],
                         mode = 'lines+markers', name = 'cv+dw',
                         line = dict(color = 'green', width = 3, dash = 'dot'),
                         marker = dict(size = 8)))

# Set the layout
fig.update_layout(title = "Vector Addition in 2D",
                  xaxis = dict(range=[0, 5], zeroline = True, title = 'X'),
                  yaxis = dict(range=[0, 5], zeroline = True, title = 'Y'),
                  showlegend = True,
                  width=600,
                  height=600)

# Show the plot
fig.show()


# In[2]:




