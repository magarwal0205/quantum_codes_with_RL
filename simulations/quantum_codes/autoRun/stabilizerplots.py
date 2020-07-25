#!/usr/bin/env python
# coding: utf-8

# In[23]:

import numpy as np
import scipy as sp
import scipy.linalg
import cvxpy as cp
import matplotlib.pyplot as plt
import random
import math
import sys


# change simulation modes here
# errType = 0 # 0 or 1
# codeType = 2 # 0, 1, 2, or 3
errType=int(sys.argv[1])
codeType=int(sys.argv[2])
tot_probabs = 10
tot_iter = 1000

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


NormalizeState = lambda state: state / sp.linalg.norm(state)
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


# change check matrix here
if(codeType == 0):
	checkMatrix = np.array([[1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0],
	                        [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1],
	                        [0,1,0,1,1,0,1,0, 0,0,0,0,1,1,1,1],
	                        [0,1,0,1,0,1,0,1, 0,0,1,1,0,0,1,1],
	                        [0,1,1,0,1,0,0,1, 0,1,0,1,0,1,0,1]])
elif(codeType == 1):
	checkMatrix = np.array([[0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0],
	                        [0,0,0,0,0,0,0,0,0, 1,0,1,0,0,0,0,0,0],
	                        [0,0,0,0,0,0,0,0,0, 0,0,0,1,1,0,0,0,0],
	                        [0,0,0,0,0,0,0,0,0, 0,0,0,1,0,1,0,0,0],
	                        [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,1,0],
	                        [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,0,1],
	                        [1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,0],
	                        [1,1,1,0,0,0,1,1,1, 0,0,0,0,0,0,0,0,0]])
elif(codeType == 2):
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


# ## Syndromes

# In[8]:


def fillSyndromeTable(checkRowCorrect):
    # get error corresponding to the given vector
    err = getGenerator(checkRowCorrect)

    # get syndrome of that error
    syndrVal = np.zeros(n-k,dtype='int')
    for i in range(n-k):
        syndBool = np.all(     np.abs(np.matmul(np.matmul(err, gi[i,:,:]), err.transpose()) - gi[i,:,:]) < comparingAccuracy_syndrome    )
        if syndBool == True:
            syndrVal[i] = 1
        else:
            syndrVal[i] = -1

    # convert syndrome to index
    syndrIndex = int(''.join(['1' if x else '0' for x in syndrVal==1]), 2)
    
    # if not already updated, update the syndrome table
    if isFilledTable[syndrIndex] == 0:
        errorRecoveryList[syndrIndex, :, :] = err
        isFilledTable[syndrIndex] = 1 


# #### Weight 1 errors:

# In[9]:


isFilledTable = np.zeros(2**(n-k))
errorRecoveryList = np.zeros([2**(n-k), 2**n, 2**n])
string_format = '{:0>' + str(n) + '}'
totErrChecked = 0

fillSyndromeTable(np.zeros(2*n, dtype = 'int'))
totErrChecked = totErrChecked + 1

myIndex = 1
while(myIndex < 2**(n)):
    # generate weight 1 vectors
    checkRow = list(string_format.format("{:b}".format(myIndex)))
    checkRow = list(map(int, checkRow))
    
    # weight 1 error with X, then Y and then Z 
    fillSyndromeTable(np.append(np.asarray(checkRow), np.zeros(n, dtype = 'int')))
    fillSyndromeTable(np.append(np.zeros(n, dtype = 'int'), np.asarray(checkRow)))
    fillSyndromeTable(np.append(np.asarray(checkRow), np.asarray(checkRow)))
    
    totErrChecked = totErrChecked + 3
        
    myIndex = myIndex*2   


# In[10]:


print(str(np.sum(isFilledTable == 1)) + ' entries filled out of total ' + str(isFilledTable.shape[0]) + ' syndromes, number of errors checked = ' + str(totErrChecked))


# #### Weight 2 errors:

# In[11]:


myIndex1 = 1
while(myIndex1 < 2**(n)):
    # generate weight 1 vectors
    checkRow1 = list(string_format.format("{:b}".format(myIndex1)))
    checkRow1 = np.asarray(list(map(int, checkRow1)))
    
    myIndex2 = myIndex1*2
    while(myIndex2 < 2**n):
        # generate another weight 1 vector
        checkRow2 = list(string_format.format("{:b}".format(myIndex2)))
        checkRow2 = np.asarray(list(map(int, checkRow2)))
        
        #generate weight 2 vector
        checkRow3 = list(string_format.format("{:b}".format(myIndex2)))
        checkRow3 = np.asarray(list(map(int, checkRow3)))
        checkRow3[checkRow1 == 1] = 1
        
        # add weight 2 errors with XX, XY, XZ, ...
        fillSyndromeTable(np.append(checkRow3, np.zeros(n, dtype = 'int')))
        fillSyndromeTable(np.append(checkRow3, checkRow1))
        fillSyndromeTable(np.append(checkRow2, checkRow1))
        
        fillSyndromeTable(np.append(checkRow3, checkRow2))
        fillSyndromeTable(np.append(checkRow3, checkRow3))
        fillSyndromeTable(np.append(checkRow2, checkRow3))
        
        fillSyndromeTable(np.append(checkRow1, checkRow2))
        fillSyndromeTable(np.append(checkRow1, checkRow3))      
        fillSyndromeTable(np.append(np.zeros(n, dtype = 'int'), checkRow3))
        
        totErrChecked = totErrChecked + 9
        myIndex2 = myIndex2*2
        
    myIndex1 = myIndex1*2   


# In[12]:


print(str(np.sum(isFilledTable == 1)) + ' entries filled out of total ' + str(isFilledTable.shape[0]) + ' syndromes, number of errors checked = ' + str(totErrChecked))


# #### Weight 3 errors:

# In[13]:


myIndex1 = 1
tempcount = 0
tempc = 0
while(myIndex1 < 2**(n)):
    # generate weight 1 vectors
    checkRowList = np.zeros([n, 4])
    
    checkRow1 = list(string_format.format("{:b}".format(myIndex1)))
    checkRow1 = np.asarray(list(map(int, checkRow1)))
    
    checkRowCombined1 = list(string_format.format("{:b}".format(myIndex1)))
    checkRowCombined1 = np.asarray(list(map(int, checkRowCombined1)))
    
    myIndex2 = myIndex1*2
    while(myIndex2 < 2**n):
        # generate another weight 1 vector
        checkRow2 = list(string_format.format("{:b}".format(myIndex2)))
        checkRow2 = np.asarray(list(map(int, checkRow2)))
        
        checkRowCombined2 = list(string_format.format("{:b}".format(myIndex2)))
        checkRowCombined2 = np.asarray(list(map(int, checkRowCombined2)))
        
        myIndex3 = myIndex2*2
        while(myIndex3 < 2**n):
            # generate another weight 1 vector
            checkRow3 = list(string_format.format("{:b}".format(myIndex3)))
            checkRow3 = np.asarray(list(map(int, checkRow3)))
            
            # generate weight 2 and 3 vectors
            checkRowCombined3 = list(string_format.format("{:b}".format(myIndex3)))
            checkRowCombined3 = np.asarray(list(map(int, checkRowCombined3)))
            
            checkRowCombined4 = list(string_format.format("{:b}".format(myIndex3)))
            checkRowCombined4 = np.asarray(list(map(int, checkRowCombined4)))
            
            checkRowCombined1[checkRow2 == 1] = 1
            checkRowCombined2[checkRow3 == 1] = 1
            checkRowCombined3[checkRow1 == 1] = 1
            
            checkRowCombined4[checkRow2 == 1] = 1
            checkRowCombined4[checkRow1 == 1] = 1
            
            fillSyndromeTable(np.append(checkRowCombined4, np.zeros(n, dtype = 'int')))
            fillSyndromeTable(np.append(checkRowCombined4, checkRow1))
            fillSyndromeTable(np.append(checkRowCombined4, checkRow2))
            fillSyndromeTable(np.append(checkRowCombined4, checkRow3))
            fillSyndromeTable(np.append(checkRowCombined4, checkRowCombined1))
            fillSyndromeTable(np.append(checkRowCombined4, checkRowCombined2))
            fillSyndromeTable(np.append(checkRowCombined4, checkRowCombined3))
            fillSyndromeTable(np.append(checkRowCombined4, checkRowCombined4))
            fillSyndromeTable(np.append(checkRowCombined3, checkRowCombined4))
            fillSyndromeTable(np.append(checkRowCombined2, checkRowCombined4))
            fillSyndromeTable(np.append(checkRowCombined1, checkRowCombined4))
            fillSyndromeTable(np.append(checkRow3, checkRowCombined4))
            fillSyndromeTable(np.append(checkRow2, checkRowCombined4))
            fillSyndromeTable(np.append(checkRow1, checkRowCombined4))
            fillSyndromeTable(np.append(np.zeros(n, dtype = 'int'), checkRowCombined4))
            
            fillSyndromeTable(np.append(checkRowCombined3, checkRowCombined2))
            fillSyndromeTable(np.append(checkRowCombined3, checkRowCombined1))
            fillSyndromeTable(np.append(checkRowCombined3, checkRow2))
            fillSyndromeTable(np.append(checkRowCombined2, checkRowCombined3))
            fillSyndromeTable(np.append(checkRowCombined1, checkRowCombined3))
            fillSyndromeTable(np.append(checkRow2, checkRowCombined3))
            
            fillSyndromeTable(np.append(checkRowCombined2, checkRowCombined1))
            fillSyndromeTable(np.append(checkRowCombined2, checkRow1))
            fillSyndromeTable(np.append(checkRowCombined1, checkRowCombined2))
            fillSyndromeTable(np.append(checkRow1, checkRowCombined2))
            
            fillSyndromeTable(np.append(checkRowCombined1, checkRow3))
            fillSyndromeTable(np.append(checkRow3, checkRowCombined1))

            totErrChecked = totErrChecked + 27
            if(np.sum(isFilledTable == 1) == isFilledTable.shape[0]):
                break
            myIndex3 = myIndex3*2
        
        if(np.sum(isFilledTable == 1) == isFilledTable.shape[0]):
            break
        myIndex2 = myIndex2*2
    
    if(np.sum(isFilledTable == 1) == isFilledTable.shape[0]):
        break
    myIndex1 = myIndex1*2 


# In[14]:


print(str(np.sum(isFilledTable == 1)) + ' entries filled out of total ' + str(isFilledTable.shape[0]) + ' syndromes, number of errors checked = ' + str(totErrChecked))


# ## Encoding

# In[15]:


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

# In[16]:


Gmatrix = np.eye(gi[0,:,:].shape[0], gi[0,:,:].shape[1]) # generator matrix corresponding to this code
for i in range(n-k):
    Gmatrix = Gmatrix + np.matmul(gi[i,:,:], Gmatrix)
Gmatrix = np.round(Gmatrix)


# #### Get non-zero and unique columns of G

# In[17]:


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


# In[18]:


print('Rank of G = ' + str(np.linalg.matrix_rank(Gmatrix)))
print('Shape of G = ' + str(Gmatrix.shape))
print('Code = ' + str(n) + ',' + str(n-k))


# In[19]:


# syndrome lookup table
def SyndromeLookUp(syndr):
    errorSyndromeIndex = int(''.join(['1' if x else '0' for x in np.ndarray.astype( np.round(syndr.flatten()), int) == 1]), 2)
    recov = errorRecoveryList[errorSyndromeIndex]
    return recov


# ## Channel and Decoding

# #### For different values of p and over many iterations

# In[ ]:

probab_list = np.linspace(0,0.33333,tot_probabs)
myError_list = np.zeros(tot_probabs)
ind_probab = 0

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
        syndr = np.zeros([n-k, 1]) # syndrome
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
                print('Warning buddy: Non zero imaginary part in the error')
            error = np.real(error_out)
        else:
            errString = 'Fidelity distance'
            errSaveName = 'fidelity'
            rho = np.outer(tx_decoded, tx_decoded)
            sigma = np.outer(rx_decoded, rx_decoded)
            error_out = np.trace(sp.linalg.sqrtm(sp.linalg.sqrtm(rho)@sigma@sp.linalg.sqrtm(rho)))
            if(np.imag(error_out) > 1e-5):
                print('Warning buddy: Non zero imaginary part in the error')
            error = np.real(error_out)          
        
        if(math.isnan(error) == False):
            myError = myError + error
        else:
            tot_iter_temp = tot_iter_temp - 1
    myError = myError/tot_iter_temp
    print('p = ' + str(p_xyz) + ', ' + errString + ' = ' + str(myError))
    myError_list[ind_probab] = myError
    ind_probab = ind_probab + 1


# In[ ]:


plt.plot(probab_list, myError_list)
plt.ylabel(errString)
plt.xlabel('p')
plt.title(errString + ' for ' + str(n) + ',' + str(k) + ' code from Gottesman')
plt.grid()
plt.tight_layout()

saveName = './plots/' + errSaveName + '_' + str(n) + ',' + str(k) + '_iter_' + str(tot_iter) + '_totp_' + str(tot_probabs) + '.png'
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
