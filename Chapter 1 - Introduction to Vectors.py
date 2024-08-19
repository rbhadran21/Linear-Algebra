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


# ### Vectors in 3 - Dimensions
# The principles mentioned earlier are the same for vectors with 3 components.<br>
# $\begin{align}
# \vec{v} = 
# \begin{bmatrix}
# v1 \\
# v2 \\
# v3\end{bmatrix}   
# \end{align}$ 
# 
# Vector Addition example:
# 

# In[55]:


v = np.array([1, 1, -1])
w = np.array([2, 3, 4])

print("V + W =",v+w)

fig = go.Figure()

fig.add_trace(go.Scatter3d(x = [0, v[0]],
                         y = [0, v[1]],
                         z = [0, v[2]],
                         mode = "lines+markers",
                         marker = dict(size = 4),
                         line = dict(color = 'red', width = 4),
                         name = "v"))

fig.add_trace(go.Scatter3d(x = [0, -v[0]],
                         y = [0, -v[1]],
                         z = [0, -v[2]],
                         mode = "lines+markers",
                         marker = dict(size = 4),
                         line = dict(color = 'red', width = 4, dash = "dot"),
                         name = "-v"))

fig.add_trace(go.Scatter3d(x = [0, w[0]],
                         y = [0, w[1]],
                         z = [0, w[2]],
                         mode = "lines+markers",
                         marker = dict(size = 4),
                         line = dict(color = 'blue', width = 4),
                         name = "w"))

fig.add_trace(go.Scatter3d(x = [0, -w[0]],
                         y = [0, -w[1]],
                         z = [0, -w[2]],
                         mode = "lines+markers",
                         marker = dict(size = 4),
                         line = dict(color = 'blue', width = 4, dash = "dot"),
                         name = "-w"))

fig.update_layout(
    scene = dict(
        xaxis = dict(range=[-6,6], title = "X"),
        yaxis = dict(range=[-6,6], title = "Y"),
        zaxis = dict(range=[-6,6], title = "Z")
        ),
    title = "3-D Vectors V & W and their inverse in 3-D Space"
)


# ### Worked out examples and select questions from the textbook

# 1. The linear combinations of $\vec{v}$ = (1, 1, 0) and $\vec{w}$ = (0, 1, 1) fill a plane. Describe that plane. Find a vector that is not a combination of $\vec{v}$ and $\vec{w}$. 
# 
# $\vec{v}$ = (1,1,0)
# $\vec{w}$ = (0,1,1)
# 
# Linear Combination with $c$ & $d$ = $c\vec{v} + d\vec{w}$ = ($c$, $c+d$, $d$)
# 
# Given that this linear combination fills a plane, let $\vec{u}$ be one of the vectors that fills the plane.
# 
# $\begin{align}
# \vec{u} = 
# \begin{bmatrix}
# c \\
# c + d \\
# d
# \end{bmatrix}
# \end{align}$
# 
# From this vector definition we can see that $u2$ is defined by the sum of $u1$ & $u3$. Any vector which doesnt follow this rule is not on the plane.
# 

# In[78]:


v = np.array([1, 1, 0])
w = np.array([0, 1, 1])

def linear_combination(c, d):
    return c * v + d * w

def is_in_plane(vector):
    x, y, z = vector
    return np.isclose(y, x + z)

def vector_in_place(u1, u2, u3):
    u = np.array([u1, u2, u3])
    in_plane = is_in_plane(u)
    print("Is vector u = ", u, " on the plane?", in_plane)
    
vector_in_place(1,2,3)
vector_in_place(1,2,1)
vector_in_place(1,2,0)


# 2. Find two equations for the unknowns c and d so that the linear combination $c\vec{v}$ + $d\vec{w}$ equals the vector $\vec{b}$: 
# 
# $\begin{align}
# \vec{v} = 
# \begin{bmatrix}
# 2 \\
# -1 \end{bmatrix}\end{align}$
# 
# $\begin{align}
# \vec{w} = 
# \begin{bmatrix}
# -1 \\
# 2 
# \end{bmatrix}
# \end{align}$
# 
# $\begin{align}
# \vec{b} = 
# \begin{bmatrix}
# 1 \\
# 0 
# \end{bmatrix}
# \end{align}$
# 
# $\vec{v}$ & $\vec{w}$ can be rewritten as a 2 x 2 matrix $A$ and $x$ would be the 2 x 1 matrix of the scalars $c$ & $d$. The matrix multiplication of the two would give then resultant vector $\vec{b}$.
# 
# $\begin{align}
# Ax = \vec{b}
# \end{align}$
# 
# $\begin{align}
# \begin{bmatrix}
# 2 & -1 \\
# -1 & 2
# \end{bmatrix} .
# \begin{bmatrix}
# c \\
# d
# \end{bmatrix}= 
# \begin{bmatrix}
# 1 \\
# 0
# \end{bmatrix}
# \end{align}$

# In[99]:


A = np.array([[2, -1],
              [-1, 2]])

b = np.array([1, 0])

print("Solution = ",
      
      np.linalg.solve(A, b))


# In[ ]:




