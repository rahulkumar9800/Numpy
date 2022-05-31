#!/usr/bin/env python
# coding: utf-8

# # NumPy

# In[1]:


import numpy as np


# In[4]:


a= np.array([1,2,3])
a


# In[5]:


a[0]


# In[6]:


a[1]


# In[7]:


import time
import sys


# In[11]:


b=range(1000)
print(sys.getsizeof(5)*len(b))


# In[14]:


c = np.arange(1000)
print(c.size*c.itemsize)


# In[15]:


size=100000


# In[19]:


L1 = range(size)
L2 = range(size)
A1 = np.arange(size)
A2 = np.arange(size)


# In[20]:


start = time.time()
result = [(x+y) for x,y in zip(L1,L2)]
result = [(x+y) for x,y in zip(L1,L2)]
print(result)
print("python list look:",(time.time()-start )*1000)


# In[21]:


start = time.time()
result = A1+A2
print("numpy array look:", (time.time()-start)*1000)


# In[30]:


a = np.array([[1,2],[3,4],[5,6]])
a


# In[25]:


a.ndim


# In[26]:


a.itemsize


# In[27]:


a.shape


# In[29]:


a = np.array([[1,2],[3,4],[5,6]], dtype=np.complex)
a


# In[31]:


np.zeros((3,4))


# In[32]:


np.ones((3,4))


# In[34]:


l = range(5)
l


# In[35]:


np.arange(5)


# In[37]:


print('concatenation example')
print(np.char.add('hello','hi'),['abc','xyz'])


# In[39]:


print(np.char.multiply("Hello ", 3))


# In[42]:


print(np.char.center("Hello ", 20,fillchar='_'))


# In[45]:


print(np.char.capitalize('hello world'))


# In[46]:


print(np.char.title('how are you doing'))


# In[47]:


print(np.char.lower(["HELLO",'WORLD']))
print(np.char.lower('HELLO'))


# In[49]:


print(np.char.upper(['python','data']))
print(np.char.upper('python is easy'))


# In[50]:


print(np.char.split('are you coming to the party'))


# In[51]:


print(np.char.splitlines('hello\nhow are you?'))


# In[54]:


print(np.char.strip(['karan','qutub','akash'],'a'))


# In[55]:


print(np.char.join([':','-'],['dmy','ymd']))


# In[56]:


print(np.char.replace('He is a good dancer','is','was'))


# # Array Manipulation

# # Array Manipulation- changing shape

# In[57]:


import numpy as np

a = np.arange(9)
print('The original array:')
print(a)
print()
b=a.reshape(3,3)
print("The modified array:")
print(b)


# In[58]:


print(b.flatten())


# In[59]:


b.flatten(order="F")


# In[61]:


a =np.arange(12).reshape(4,3)
a


# In[63]:


print(np.transpose(a))


# In[67]:


b = np.arange(8).reshape(2,4)
b


# In[70]:


c = b.reshape(2,2,2)
c


# In[72]:


np.rollaxis(c,1,2)


# # Numpy Arithematic operations

# In[73]:


a = np.arange(9).reshape(3,3)
a


# In[82]:


b = np.array([10,10,10])
b


# In[81]:


np.add(a,b)


# In[77]:


np.subtract(a,b)


# In[78]:


np.multiply(a,b)


# In[79]:


np.divide(a,b)


# # Slicing

# In[83]:


a  =np.arange(20)
a


# In[86]:


a[:4]


# In[87]:


a[5]


# In[91]:


s = slice(2,12,3)
a[s]


# # Iteration Over Array

# In[96]:


a = np.arange(0,45,5 )
a = a.reshape(3,3)
a


# In[97]:


for x in np.nditer(a):
    print(x)


# # Iteration Order (c-style and f-style)

# In[100]:


print(a)
for x in np.nditer(a, order="C"):
    print(x)
    
for x in np.nditer(a, order="F"):
    print(x)


# # Joining Array

# In[107]:


a = np.array([[1,2,],[3,4]])
print('first array')
print(a)
b = np.array([[5,6],[7,8]])
print('second array')
print(b)
print('\n')
print('joining the two arrays along axis 0:')
print( np.concatenate((a,b)))
print('\n')
print('joining the two arrays along axis 1:')
print(np.concatenate((a,b),axis = 1))


# # Spliting array

# In[110]:


a = np.arange(9)
print(a)
np.split(a,3)


# In[113]:


np.split(a,[4,5,7])


# In[112]:


np.split(a,[4,7])


# # Resizing an array

# In[117]:


a = np.array([[1,2,3],[4,5,6]])
print(a)
print(a.shape)
print('\n')
b = np.resize(a,(3,2))
print(b)
print(b.shape)
print('\n')
b =np.resize(a,(3,3))
print(b)
print(b.shape)


# # Historgram

# In[119]:


from matplotlib import pyplot as plt
a = np.array([20,87,4,40,53,74,56,51,11,20,40,15,79,25,27])
plt.hist(a, bins = [0,20,40,60,80,100])
plt.title("Histogram")
plt.show()


# In[120]:


plt.hist(a, bins = [0,10,20,40,50,60,70,80,90,100])
plt.title("Histogram")
plt.show()


# # Other useful function in numpy

# In[122]:


#linspace function

a=np.linspace(1,3,10)
print(a)


# In[130]:


#sum and axis

a= np.array([(1,2,3),(3,4,5)])
print(a.sum(axis=0))


# In[131]:


#Square root and standard deviation

a= np.array([(1,2,3),(3,4,5)])
print(np.sqrt(a))
print(np.std(a))


# In[132]:


#Ravel function

import numpy as np
x= np.array([(1,2,3),(3,4,5)])
print(x.ravel())


# In[133]:


a= np.array([1,2,3])
print(np.log10(a))


# # Numpy Practice Example

# In[139]:


#now using numpy and matplotlib, let us try to plot a sin graph
import numpy as np 
import matplotlib.pyplot as plt

x= np.arange(0,3*np.pi,0.1)
y= np.sin(x)
print(x)
print(y)
plt.plot(x,y)
plt.show()


# In[144]:


#creat a 6*6 two dimensional array,and Let 1 and 0 be placed alternatively across the diagonals
z = np.zeros((6,6),dtype=int)
z[1::2,::2]= 1
z[::2,1::2]=1
z


# In[147]:


#Find the total number and Location of missing values in the array
z= np.random.rand(10,10)
z[np.random.randint(10, size=5), np.random.randint(10, size=5)]= np.nan
z


# In[155]:


print("Total number of missing values: \n", np.isnan(z).sum())
print("Indexes of missing values: \n", np.argwhere(np.isnan(z)))
inds = np.where(np.isnan(z))
print(inds)
z[inds]=0
z


# In[ ]:




