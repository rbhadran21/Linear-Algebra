#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cosine


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

print("Solution = ", np.linalg.solve(A, b))


# 3. What combination of  
# $\begin{align}
# c\begin{bmatrix}
# 1 \\
# 2
# \end{bmatrix}+d\begin{bmatrix}
# 3 \\
# 1
# \end{bmatrix}
# \end{align}$
# produces
# $\begin{align}
# \begin{bmatrix}
# 14 \\
# 8
# \end{bmatrix}?
# \end{align}$

# In[100]:


A = np.array([[1, 2],
              [3, 1]])

b = np.array([14, 8])

print("Solution = ", np.linalg.solve(A, b))


# 4. Find vectors $\vec{v}$ and $\vec{w}$ so that $\vec{v} + \vec{w} = (4,5,6)$ and $\vec{v} - \vec{w} = (2,5,8)$
# Solution:
# 
# $\vec{v} = (v1, v2 ,v3)$<br>
# $\vec{w} = (w1, w2 ,w3)$<br>
# 
# $\vec{v} + \vec{w} = (v1+w1, v2+w2, v3+w3) = (4, 5, 6)$<br>
# $\vec{v} - \vec{w} = (v1-w1, v2-w2, v3-w3) = (2, 5, 8)$

# In[104]:


A = np.array([[1, 1],
              [1, -1]])
b1 = np.array([4, 2])
b2 = np.array([5, 5])
b3 = np.array([6, 8])

v1, w1 = np.linalg.solve(A, b1)
v2, w2 = np.linalg.solve(A, b2)
v3, w3 = np.linalg.solve(A, b3)

print('V = ', (v1,v2,v3))
print('W = ', (w1,w2,w3))


# 5. Find two different combinations of the three vectors $\vec{u} = (1,3)$ and $\vec{v} = (2, 7)$ and $\vec{w} = (1,5)$ that produce $\vec{b} = (0,1)$.

# In[125]:


A = np.array([
    [1, 2, 1],
    [3, 7, 5]
])

B = np.array([0, 1])

# Since the number of unknowns are more than knowns we have infinite solutions
solution = np.linalg.lstsq(A, B, rcond=None)[0]

print("Particular solution:", solution)

# Null space of A (to find the general solution) that can be used to find infinite solutions
null_space = np.linalg.svd(A)[2].T[:, 2]

print("Alternate solution 1:", solution + null_space)
print("Alternate solution 2:", solution + 2*null_space)


# ### Dot Products
# Dot products is a way to multiply two vectors to get a single number which helps give sense to the direction of two vectors <br>
# $\begin{align}
# A.B = ||A||.||B||cos(\theta)
# \end{align}$
# 
# * Where $||A||$ & $||B||$ are the lengths (magnitudes) of vectors $A$ & $B$<br>
# * $\theta$ is the angle between the two vectors.
# 
# ###  Length of vectors:
# The length of a vector is its magnitude, usually given by the $norm$ of the vector.<br>
# $\begin{align}length(v) = ||v|| = \sqrt{v.v}\end{align}$
# **Geometric Interpretation:<br>**
# The dot product also tells us something about the angle between two vectors:
# 
# * **Positive dot product:** The angle between the vectors is less than $90\degree$, and the vectors are pointing more in the same direction.<br>
# * **Zero dot product:** The vectors are perpendicular ($90\degree$ apart).<br>
# * **Negative dot product:** The angle between the vectors is greater than $90\degree$, and the vectors are pointing in mostly opposite directions.
# 
# ### Angles between vector:
# The angle between 2 vectors is given by the follow:<br>
# $\begin{align}\theta = \cos^{-1}\left(\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}\right)\end{align}$

# In[144]:


## Examples of dot products and geometrics interpretation
vector1 = np.array([3, 4])
vector2 = np.array([6, 8])

dot_product = np.dot(vector1, vector2)

print(f"Dot product of {vector1} and {vector2} is: {dot_product}")

fig = go.Figure()
fig.add_trace(go.Scatter(x = [0, vector1[0]],
                         y = [0, vector1[1]],
                         mode = 'lines+markers',
                         line = dict(color = "blue"),
                         name = "vector1"))

fig.add_trace(go.Scatter(x = [0, vector2[0]],
                         y = [0, vector2[1]],
                         mode = 'lines+markers',
                         line = dict(color = "blue"),
                         name = "vector2"))

vector3 = np.array([3, 0])
vector4 = np.array([0, 4])

dot_product_perpendicular = np.dot(vector3, vector4)

print(f"Dot product of {vector3} and {vector4} is: {dot_product_perpendicular}")

fig.add_trace(go.Scatter(x = [0, vector3[0]],
                         y = [0, vector3[1]],
                         mode = 'lines+markers',
                         line = dict(color = "red"),
                         name = "vector3"))

fig.add_trace(go.Scatter(x = [0, vector4[0]],
                         y = [0, vector4[1]],
                         mode = 'lines+markers',
                         line = dict(color = "red"),
                         name = "vector4"))

vector5 = np.array([2, 3])
vector6 = np.array([-2, -3])

dot_product_opposite = np.dot(vector5, vector6)

print(f"Dot product of {vector5} and {vector6} is: {dot_product_opposite}")

fig.add_trace(go.Scatter(x = [0, vector5[0]],
                         y = [0, vector5[1]],
                         mode = 'lines+markers',
                         line = dict(color = "green"),
                         name = "vector5"))

fig.add_trace(go.Scatter(x = [0, vector6[0]],
                         y = [0, vector6[1]],
                         mode = 'lines+markers',
                         line = dict(color = "green"),
                         name = "vector6"))

fig.update_layout(xaxis = dict(range = [-6, 6]),
                  yaxis = dict(range = [-6, 6]),
                  width = 600,
                  height = 800)

fig.show()


# In[157]:


def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2) ## calculating A.B
    norm_vector1 = np.linalg.norm(vector1) ## calculating length(A)
    norm_vector2 = np.linalg.norm(vector2) ## calculating length(B)
    cos_theta = dot_product / (norm_vector1 * norm_vector2) 
    angle_radians = np.arccos(cos_theta)   ## theta is returned in radians
    angle_degrees = np.degrees(angle_radians) ## converting radians in degree (multipley by 57.32 or 180/pi)
    
    return angle_degrees

vector1 = np.array([1, 2])
vector2 = np.array([1, 3])

angle = angle_between_vectors(vector1, vector2)

print(f"The angle between {vector1} and {vector2} is: {angle:.2f} degrees")

