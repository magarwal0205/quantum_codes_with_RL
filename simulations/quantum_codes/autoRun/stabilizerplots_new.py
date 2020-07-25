#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import scipy.linalg
import cvxpy as cp
import matplotlib.pyplot as plt
import random
import math
import sys


errType=int(sys.argv[1])
codeType=int(sys.argv[2])
tot_iter=int(sys.argv[3])
tot_probabs = 10
# In[2]:


def NKron(*args):
  result = np.array([[1.0]])
  for op in args:
    result = np.kron(result, op)
  return result


# In[3]:


Id = np.eye(2)
X = np.array([[0.0, 1.0],[1.0, 0.0]])
Z = np.array([[1.0, 0.0],[0.0, -1.0]])
Y = np.matmul(X,Z)


# In[4]:


# NormalizeState = lambda state: state / sp.linalg.norm(state)
def NormalizeState(ipVal):
    if(sp.linalg.norm(ipVal) == 0): return ipVal
    else : return ipVal / sp.linalg.norm(ipVal)
zero = np.array([[1.0], [0.0]]) # |0>
one = np.array([[0.0], [1.0]]) # |1>


# ## Generators

# In[5]:


def NKronModified(checkRowMod):
  result = np.array([[1.0]])
  for ind in checkRowMod:
    if(ind == 0):
        op = Id
    elif(ind == 1):
        op = X
    elif(ind == 2):
        op = Y
    elif(ind == 3):
        op = Z
    result = np.kron(result, op)
  return result

def getGenerator(checkRow):
    checkRowModified = np.zeros(n, dtype=int)
    
    checkRowModified[(checkRow[:n] == checkRow[n:]) & (checkRow[n:] == 1)] = 2
    checkRowModified[(checkRow[:n] == 1) & (checkRowModified != 2)] = 1
    checkRowModified[(checkRow[n:] == 1) & (checkRowModified != 2)] = 3
    
    return NKronModified(checkRowModified)    


# In[6]:


comparingAccuracy_decoded = 1e-7
comparingAccuracy_encoded = 1e-5
comparingAccuracy_syndrome = 1e-5
comparingAccuracy_method = 1e-5


# In[7]:


Hmatrix = np.array([[1,0,1,0,1,0,1],
                    [0,1,1,0,0,1,1],
                    [0,0,0,1,1,1,1]])

na = Hmatrix.shape[1]
nb = Hmatrix.shape[0]

Hx = np.concatenate((np.kron(np.eye(na), Hmatrix), np.kron(Hmatrix.T, np.eye(nb))), axis=1)
Hz = np.concatenate((np.kron(Hmatrix, np.eye(na)), np.kron(np.eye(nb), Hmatrix.T)), axis=1)

H1 = np.concatenate((Hx, np.zeros([Hx.shape[0], Hx.shape[1]])), axis=1)
H2 = np.concatenate((np.zeros([Hz.shape[0], Hz.shape[1]]), Hz), axis=1)

# checkMatrix = np.concatenate((H1, H2), axis=0)


# In[8]:


# change check matrix here

if(codeType == 0):
    checkMatrix = np.array([[1,0,0,1,0, 0,1,1,0,0],
                            [0,1,0,0,1, 0,0,1,1,0],
                            [1,0,1,0,0, 0,0,0,1,1],
                            [0,1,0,1,0, 1,0,0,0,1]])
else:
    checkMatrix = np.array([[0,0,0,1,1,1,1, 0,0,0,0,0,0,0],
                            [0,1,1,0,0,1,1, 0,0,0,0,0,0,0],
                            [1,0,1,0,1,0,1, 0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0, 0,0,0,1,1,1,1],
                            [0,0,0,0,0,0,0, 0,1,1,0,0,1,1],
                            [0,0,0,0,0,0,0, 1,0,1,0,1,0,1]])

n = int(checkMatrix.shape[1]/2)
k = n-checkMatrix.shape[0]

gi = np.zeros([n-k, 2**n, 2**n])
for i in range(n-k):
    gi[i,:,:] = getGenerator(checkMatrix[i,:])
    
# def gi(i):
#     return getGenerator(checkMatrix[i,:])


# ## Encoding

# In[9]:


def NKron1DGeneral(ipArray):
    result = np.array([[1.0]])
    for i in ipArray:
        if(i==1):
            op = one
        elif(i==0):
            op = zero
        result = np.kron(result, op)
    return result


# #### Get generator matrix G

# In[10]:


Gmatrix = np.eye(gi[0,:,:].shape[0], gi[0,:,:].shape[1]) # generator matrix corresponding to this code
for i in range(n-k):
    Gmatrix = Gmatrix + np.matmul(gi[i,:,:], Gmatrix)
Gmatrix = np.round(Gmatrix)


# #### Get non-zero and unique columns of G

# In[11]:


# get boolean array if the columns are zero or not
zeroCols = np.zeros(Gmatrix.shape[1])
for i in range(Gmatrix.shape[1]):
    zeroCols[i] = all(Gmatrix[:,i] == np.zeros(Gmatrix.shape[0]))

# get indices of non-zero columns
nonZeroColsList = np.argwhere(zeroCols==0).flatten()

# get all non zero columns
GmatrixNonZero = np.zeros([Gmatrix.shape[0], nonZeroColsList.shape[0]])
i = 0
for ind in nonZeroColsList:
    GmatrixNonZero[:,i] = Gmatrix[:,ind]
    i = i+1

# get all non zero and unique columns and there indices
GmatrixNonZeroUniqueInd, nonZeroUniqueInd = np.unique(GmatrixNonZero, axis = 1, return_index=True)
nonZeroUniqueInd = nonZeroColsList[nonZeroUniqueInd]


# In[12]:



# In[13]:


def getSyndromeFromError(channel_error):
    tx_qbits = np.ones(2**k)
    tx_qbits = NormalizeState(tx_qbits)

    # Convert qbits to tensor product format
    tx_decoded = np.zeros(2**n)
    # get extended qbits corresponding to non-zero column indices of G matrix
    i = 0
    for nonZeroIndex in np.sort(nonZeroUniqueInd):
        if(i>=2**k):
            break
        tx_decoded[nonZeroIndex] = tx_qbits[i]
        i = i+1
    tx_decoded = NormalizeState(tx_decoded)

    # encode transmit qbits
    tx_encoded = NormalizeState(tx_decoded) # encoded transmit qbits
    for i in range(n-k):
        tx_encoded = tx_encoded + np.matmul(gi[i,:,:], tx_encoded) # encode using generators
    tx_encoded = NormalizeState(tx_encoded) # encoded transmit qbits

    # channel
    rx_erry = np.dot(channel_error, tx_encoded) # received qbits with errors

    # syndrome check
    syndr = np.zeros([n-k, 1]) # syndrome
    for i in range(n-k):
        syndr[i] = np.dot(rx_erry.transpose(), np.dot(gi[i,:,:], rx_erry))
        
    syndr[syndr>0] = 0
    syndr[syndr<0] = 1
        
    return np.ndarray.astype(np.round(syndr), 'int').flatten()


# In[14]:



# In[15]:


def getCardin(myVector):
    return np.sum(myVector != 0)

def getErrorFromSyndrome(syndr):
    success = 0
    finalError = np.zeros(2*errCheckRowModified.shape[0])
    while(getCardin(syndr) != 0):
        maxMetric = 0
        for generatorInd in range(n-k): # for all generators
            g = checkMatrix[generatorInd, :] # get the genrator
            g_modified = np.zeros(n, dtype=int)
            g_modified[(g[:n] == g[n:]) & (g[n:] == 1)] = 2
            g_modified[(g[:n] == 1) & (g_modified != 2)] = 1
            g_modified[(g[n:] == 1) & (g_modified != 2)] = 3
            
            string_format = '{:0>' + str(2*getCardin(g_modified)) + '}'
            for errorIndex in range(2**(2*getCardin(g_modified))): # for all errors with the support of that generator
                if(errorIndex == 0): continue
                thisError = np.copy(g_modified)
                
                modifyError = list(string_format.format("{:b}".format(errorIndex)))
                modifyError =  np.asarray(list(map(int, modifyError)) )
                
                temp_n = getCardin(g_modified)
                modifyErrorModified = np.zeros(temp_n, dtype=int)
                modifyErrorModified[(modifyError[:temp_n] == modifyError[temp_n:]) & (modifyError[temp_n:] == 1)] = 2
                modifyErrorModified[(modifyError[:temp_n] == 1) & (modifyErrorModified != 2)] = 1
                modifyErrorModified[(modifyError[temp_n:] == 1) & (modifyErrorModified != 2)] = 3
                
                thisError[thisError != 0] = modifyErrorModified # get the error
                
                           
                thisError1 = np.copy(thisError)
                thisError1[thisError == 1] = 1
                thisError1[thisError == 2] = 1
                thisError1[thisError == 3] = 0
                
                thisError2 = np.copy(thisError)
                thisError2[thisError == 1] = 0
                thisError2[thisError == 2] = 1
                thisError2[thisError == 3] = 1
                
                thisError = np.append(thisError1, thisError2)
                
#                 print('Im here ' + str(syndr))
                syndr_new = (syndr + getSyndromeFromError(getGenerator(thisError)))%2 # update to syndrome to check weight
                thisMetric = (getCardin(syndr) - getCardin(syndr_new))/getCardin(modifyErrorModified) # get the metric
                
                if(thisMetric > maxMetric):
                    bestError = thisError
                    maxMetric = thisMetric
#                 if(thisMetric == maxMetric):
#                     print('Error = ' + str(thisError) + ', |s_i+1| = ' + str(getCardin(syndr_new)) + ', |s_i| = ' + str(getCardin(syndr)) + ', |e| = ' + str(getCardin(thisError)))

        if(maxMetric != 0):
            finalError = bestError
            syndr = (syndr + getSyndromeFromError(getGenerator(bestError)))%2
        if(maxMetric == 0):
            break        
#         print('Max metric = ' + str(maxMetric) + ', Best error = ' + str(bestError) + ', Syndrome = ' + str(syndr))

    if(getCardin(syndr) != 0): success = 0
    else: success = 1

    return finalError.flatten(), success


# In[16]:


# syndrome lookup table, but not :P
def SyndromeLookUp(syndr):
    syndr[syndr>0] = 0
    syndr[syndr<0] = 1
    error, success = getErrorFromSyndrome(syndr)
    recov = getGenerator(error)
    return recov


# ## Channel and Decoding

# #### For different values of p and over many iterations

# In[17]:


probab_list = np.linspace(0,0.33333,tot_probabs)
myError_list = np.zeros(tot_probabs)
ind_probab = 0
avgError = 0

P = np.matmul(Gmatrix.transpose(), Gmatrix)
A = np.eye(2**n, 2**n) # condition on codes
# get extended qbits corresponding to non-zero column indices of G matrix
i = 0
for nonZeroIndex in np.sort(nonZeroUniqueInd):
    if(i>=2**k):
        break
    A[nonZeroIndex, nonZeroIndex] = 0
    i = i+1

for p_xyz in probab_list:
    myError = 0
    tot_iter_temp = tot_iter
    for iter in range(tot_iter):
        # generate qbits randomly
        tx_qbits = np.random.rand(2**k)
        tx_qbits = NormalizeState(tx_qbits)

        # Convert qbits to tensor product format
        tx_decoded = np.zeros(2**n)
        # get extended qbits corresponding to non-zero column indices of G matrix
        i = 0
        for nonZeroIndex in np.sort(nonZeroUniqueInd):
            if(i>=2**k):
                break
            tx_decoded[nonZeroIndex] = tx_qbits[i]
            i = i+1
        tx_decoded = NormalizeState(tx_decoded)

        # encode transmit qbits
        tx_encoded = NormalizeState(tx_decoded) # encoded transmit qbits
        for i in range(n-k):
            tx_encoded = tx_encoded + np.matmul(gi[i,:,:], tx_encoded) # encode using generators
        tx_encoded = NormalizeState(tx_encoded) # encoded transmit qbits

        # channel
        p_channel = [1-3*p_xyz, p_xyz, p_xyz, p_xyz] 
        errMatrix = np.random.multinomial(1, p_channel, size=n)
        errCheckRowModified = errMatrix@np.array([0,1,2,3])
        channel_error = NKronModified(errCheckRowModified) # channel error
        rx_erry = np.dot(channel_error, tx_encoded) # received qbits with errors

        # syndrome check
        syndr = np.zeros(n-k) # syndrome
        for i in range(n-k):
            syndr[i] = np.dot(rx_erry.transpose(), np.dot(gi[i,:,:], rx_erry))

        # error correction
        recov = SyndromeLookUp(syndr) # error recovery
        rx_encoded = np.matmul(recov.transpose(), rx_erry) # received qbits without error but still encoded

        # complete decoding
        # setup optimizer to decode
        q = -np.matmul(rx_encoded.transpose(), Gmatrix).flatten()
        x = cp.Variable(rx_encoded.shape[0])
        # get qbit that is at closest distance to received encoded qbit
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T@x), [A@x == np.zeros(x.shape[0])])
        prob.solve()
        rx_decoded = NormalizeState(x.value) # received decoded qbits
        
        if(errType == 0):
            errString = 'Trace distance'
            errSaveName = 'trace'
            
            error = np.outer(tx_decoded, tx_decoded) - np.outer(rx_decoded, rx_decoded)
            error_out = np.trace(sp.linalg.sqrtm(error.transpose()@error))/2
            if(np.imag(error_out) > 1e-5):
                print('Warning hai: Non zero imaginary part in the error: ' + str(np.imag(error_out)))

#             error_out = tx_decoded - rx_decoded
#             error_out = np.sum(error_out**2)/np.sum(np.abs(tx_decoded) > comparingAccuracy_decoded)

#             error_out = np.abs(tx_decoded**2 - rx_decoded**2)
#             error_out = np.sum(error_out)/2

            error = np.real(error_out)
        else:
            errString = 'Fidelity distance'
            errSaveName = 'fidelity'
            rho = np.outer(tx_decoded, tx_decoded)
            sigma = np.outer(rx_decoded, rx_decoded)
            error_out = np.trace(sp.linalg.sqrtm(sp.linalg.sqrtm(rho)@sigma@sp.linalg.sqrtm(rho)))
            if(np.imag(error_out) > 1e-5):
                print('Warning hai: Non zero imaginary part in the error: ' + str(np.imag(error_out)))
            error = np.real(error_out)
        # print('i = ' + str(iter) + ', p = ' + str(p_xyz) + ', ' + errString + ' = ' + str(avgError))       
        
        if(math.isnan(error) == False):
            myError = myError + error
        else:
            tot_iter_temp = tot_iter_temp - 1
    myError = myError/tot_iter_temp
    avgError = myError
    print('p = ' + str(p_xyz) + ', ' + errString + ' = ' + str(myError))
    myError_list[ind_probab] = myError
    ind_probab = ind_probab + 1


# In[20]:


# In[19]:


plt.plot(probab_list, myError_list)
plt.ylabel(errString)
plt.xlabel('p')
plt.title(errString + ' for ' + str(n) + ',' + str(k) + ' code from Gottesman')
plt.grid()
plt.tight_layout()

saveName = './plots/new_' + errSaveName + '_' + str(n) + ',' + str(k) + '_iter_' + str(tot_iter) + '_totp_' + str(tot_probabs) + '.png'
# plt.show()
plt.savefig(saveName)
print(saveName)


# ### TODO
# - Implement minimum fidelity
# - Generate plots for other codes
# 
# 
# - Check how decoding is done in Gottesman (to fix the null-space problem)
# - Maybe try implementing the circuits if there are any
# 
# 
# - Maybe start generating data

# ## Rough

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




