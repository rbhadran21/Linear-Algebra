#!/usr/bin/env python
# coding: utf-8

# # Chapter 1: Introduction to Vectors

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

# In[2]:


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


# ### 1.1 Vectors and Linear Combinations

# ### Vector Addition<br>
# Let $v$ be a 2-dimensional vector with components 
# $\begin{align}
# \vec{v} = 
# \begin{bmatrix}
# v1 \\
# v2\end{bmatrix}
# \end{align}$
# 
# Let $w$ be a 2-dimensional vector with components 
# $\begin{align}
# \vec{w} = 
# \begin{bmatrix}
# w1 \\
# w2\end{bmatrix}
# \end{align}$
# 
# The addition of the two vectors then looks like: <br>
# $\begin{align}
# \vec{v} + \vec{w} = 
# \begin{bmatrix}
# v1 + w1 \\
# v2 + w2\end{bmatrix}
# \end{align}$
# 
# This is essentially how **vector addition** looks like.
# 
# **Subtraction** occurs follows the same principle.
# ### Scalar Multiplication<br>
# Going back to the vector $v$. If we multiply a constant $c$ to $v$ where $c$ can be any real number then we get the follow:
# $\begin{align}
# c\vec{v} = 
# \begin{bmatrix}
# cv1 \\
# cv2\end{bmatrix}
# \end{align}$
# 
# This $c$ constant is called the scalar and this operation is called **scalar multiplication**.
# 
# ### Notice that the sum of -v and v is the zero vector. This is 0, which is not the same as the number zero! The vector 0 has components 0 and 0.
# ### What does this mean?
# ### Zero Vector $\vec{0}$
# - The zero vector, often denoted as $\vec{0}$, is a vector in a vector space where all of its components are zero.
# - For example, in 2-dimensional space, the zero vector is $\vec{0} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$. In 3-dimensional space, it would be $\vec{0} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$.
# - The zero vector has both a magnitude of zero and no specific direction.
# - It is the additive identity in vector spaces, meaning for any vector $\vec{v}$, adding the zero vector results in $\vec{v}$ itself: $\vec{v} + \vec{0} = \vec{v}$.
# 
# ### Number Zero ($0$)
# - The number zero is a scalar, a real number (or a complex number, depending on the context) that represents the absence of any quantity.
# - It is the additive identity in the field of real numbers (or any other number field), meaning for any real number $x$, adding zero results in $x$: $x + 0 = x$.
# 
# ### Key Differences
# - **Nature**: The zero vector is a vector, which means it has components, and it exists in a vector space. The number zero is a scalar, existing in a number field like the real numbers.
# - **Representation**: The zero vector is written as a vector (e.g., $\begin{bmatrix} 0 \\ 0 \end{bmatrix}$ in 2D), while the number zero is written as $0$.
# - **Context of Use**: The zero vector is used in operations involving vectors, while the number zero is used in operations involving scalars.
# 
# ### Example
# 

# In[7]:


v = np.array([2,3])
w = (-1)*v
print("v + (-v)",v + w)

fig = go.Figure()

fig.add_trace(go.Scatter(x = [0, v[0]],
                         y = [0, v[1]],))

fig.add_trace(go.Scatter(x = [0, w[0]],
                         y = [0, w[1]],))

fig.update_layout(width = 600,
                  height = 600)


# This result is the zero vector $\vec{0}$, not the number zero.<br>
# This is also a good way to show that it doesnt really matter how you add 2 vectors they will result in the same output.

# In[30]:


v = np.array([4, 2])
w = np.array([-1, 2])
v_1 = (-1)*v
w_1 = (-1)*w
sum_vw = v + w
diff_vw = v - w

fig = go.Figure()

fig.add_trace(go.Scatter(x = [0, v[0]],
                         y = [0, v[1]],
                         mode = "lines+markers",
                         name = "v",
                         line = dict(color = "grey")))

fig.add_trace(go.Scatter(x = [v[0], sum_vw[0]],
                         y = [v[1], sum_vw[1]],
                         name = "w'",
                         mode = "lines+markers",
                         line = dict(color = "grey", dash = 'dot')))

fig.add_trace(go.Scatter(x = [0, w[0]],
                         y = [0, w[1]],
                         mode = "lines+markers",
                         name = "w",
                         line = dict(color = "grey")))

fig.add_trace(go.Scatter(x = [w[0], sum_vw[0]],
                         y = [w[1], sum_vw[1]],
                         mode = "lines+markers",
                         name = "v'",
                         line = dict(color = "grey", dash = 'dot')))

fig.add_trace(go.Scatter(x = [0, v_1[0]],
                         y = [0, v_1[1]],
                         mode = "lines+markers",
                         name = "-v",
                         line = dict(color = "grey", dash ="dot")))

fig.add_trace(go.Scatter(x = [0, w_1[0]],
                         y = [0, w_1[1]],
                         mode = "lines+markers",
                         name = "-w",
                         line = dict(color = "grey", dash ="dot")))

fig.add_trace(go.Scatter(x = [0, sum_vw[0]],
                         y = [0, sum_vw[1]],
                         mode = "lines+markers",
                         name = "v+w",
                         line = dict(color = "grey")))

fig.add_trace(go.Scatter(x = [0, diff_vw[0]],
                         y = [0, diff_vw[1]],
                         mode = "lines+markers",
                         name = "v-w",
                         line = dict(color = "grey", dash ="dot")))

fig.update_layout(xaxis = dict(range = [-6,6], title = "X"),
                  yaxis = dict(range = [-6,6], title = "Y"),
                  width = 800,
                  height = 800)

