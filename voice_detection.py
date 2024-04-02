#!/usr/bin/env python
# coding: utf-8

# # Voice Gender Detection Model

# ### Importing the dataset 
# 
# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


csv_data = pd.read_csv('D:\\Datasets\\train_audio.csv')
test=pd.read_csv("D:\\Datasets\\valid-test.csv")


# In[3]:


test.head()


# In[4]:


csv_data.dtypes


# In[5]:


csv_data.head(7)


# #### dropping null values

# In[6]:


data=csv_data[csv_data['gender'].notna()]


# #### dropping unecessary columns and reseting index after dropping 

# In[7]:


data=data.drop(['up_votes','down_votes','duration','accent','text'],axis=1)


# In[8]:


data.reset_index(inplace = True)


# In[9]:


data=data.drop('index',axis=1)


# In[10]:


test['gender'].isnull().sum()
test_data=test[test['gender'].notna()]
test_data.head(11)
test_data=test_data.drop(['up_votes','down_votes','duration','accent','text'],axis=1)
test_data.reset_index(inplace = True)
test_data=test_data.drop('index',axis=1)
test_data


# In[11]:


test_files=test_data.filename

test_f=test_files.tolist()
sliced_test=[]
for m in test_f:
    
    sl=slice(14,31)
    test=m[sl]
    sliced_test.append(test)

print(sliced_test)


# In[12]:


male_df=data[data['gender']=='male']
male_df_subset=male_df.head(13855)
female_df=data[data['gender']=='female']


# #### slicing the name of the file and keeping what we require to use further
# 
# 

# In[13]:


male_files=male_df_subset.filename
female_files=female_df.filename


# In[14]:


male=male_files.tolist()
sliced_male=[]
for m in male:
    
    sl=slice(15,32)
    mal=m[sl]
    sliced_male.append(mal)


# In[15]:


female=female_files.tolist()
sliced_female=[]
for f in female:
    sl= sl=slice(15,32)
    fem=f[sl]
    sliced_female.append(fem)


# ### giving access to the audio files
# 

# In[16]:


from glob import glob
audio_file=glob('C:/Program Files (x86)/audio_files/cv-other-train/*.mp3')
test_audio=glob('C:/Program Files (x86)/audio_files/cv-valid-test/*.mp3')


# In[17]:


for file in audio_file:
       sl=slice(50,68)
       
       path=file[sl]
       print(path)


# #### Function to extract Male MFCC features 

# In[19]:


def male_convert():
   import librosa
   from python_speech_features import mfcc
   from sklearn import preprocessing 
   from scipy.signal.windows import hann
  
   for file in audio_file:
       sl=slice(50,68)
       
       path=file[sl]
       
       if path in sliced_male:
           print(path)
           n_fft=512
        
           hop_len=160
           n_mels=40
           fmin=0
           fmax=None
           n_mfcc=13
           audio,sr=librosa.load(file)
           features = mfcc(# The audio signal from which to compute features.
                           audio,
                           # The samplerate of the signal we are working with.
                           sr,
                           # The length of the analysis window in seconds. 
                           # Default is 0.025s (25 milliseconds)
                           winlen       = 0.05,
                           # The step between successive windows in seconds. 
                           # Default is 0.01s (10 milliseconds)
                           winstep      = 0.01,
                           # The number of cepstrum to return. 
                           # Default 13.
                           numcep       = 2,
                           # The number of filters in the filterbank.
                           # Default is 26.
                           nfilt        = 30,
                           # The FFT size. Default is 512.
                           nfft         = 2048,
                           # If true, the zeroth cepstral coefficient is replaced 
                           # with the log of the total frame energy.
                           appendEnergy = True)
           mfcc_feature  = preprocessing.scale(features)
           print('ouput')
           return mfcc_feature
       
     
      
       
          


# #### Function to extract Female MFCC features 

# In[20]:


def female_convert():
    import librosa
    from python_speech_features import mfcc
    from sklearn import preprocessing 
    from scipy.signal.windows import hann
    for file in audio_file:
        sl=slice(50,68)
        
        path=file[sl]
        if path in sliced_female:
            n_fft=512
            
            hop_len=160
            n_mels=40
            fmin=0
            fmax=None
            n_mfcc=13
            audio,sr=librosa.load(file)
            features = mfcc(# The audio signal from which to compute features.
                            audio,
                            # The samplerate of the signal we are working with.
                            sr,
                            # The length of the analysis window in seconds. 
                            # Default is 0.025s (25 milliseconds)
                            winlen       = 0.05,
                            # The step between successive windows in seconds. 
                            # Default is 0.01s (10 milliseconds)
                            winstep      = 0.01,
                            # The number of cepstrum to return. 
                            # Default 13.
                            numcep       = 2,
                            # The number of filters in the filterbank.
                            # Default is 26.
                            nfilt        = 30,
                            # The FFT size. Default is 512.
                            nfft         = 2048,
                            # If true, the zeroth cepstral coefficient is replaced 
                            # with the log of the total frame energy.
                            appendEnergy = True)     
            mfcc_feature  = preprocessing.scale(features)
            print('ouput')
            return mfcc_feature
       


# #### Function to extract test input features 

# In[21]:


def input_convert():
    import librosa
    from python_speech_features import mfcc
    from sklearn import preprocessing 
    from scipy.signal.windows import hann
    for file in test_audio:
        n_fft=512
      
        hop_len=160
        n_mels=40
        fmin=0
        fmax=None
        n_mfcc=13
        audio,sr=librosa.load(file)
        features = mfcc(# The audio signal from which to compute features.
                            audio,
                            # The samplerate of the signal we are working with.
                            sr,
                            # The length of the analysis window in seconds. 
                            # Default is 0.025s (25 milliseconds)
                            winlen       = 0.05,
                            # The step between successive windows in seconds. 
                            # Default is 0.01s (10 milliseconds)
                            winstep      = 0.01,
                            # The number of cepstrum to return. 
                            # Default 13.
                            numcep       = 2,
                            # The number of filters in the filterbank.
                            # Default is 26.
                            nfilt        = 30,
                            # The FFT size. Default is 512.
                            nfft         = 2048,
                            # If true, the zeroth cepstral coefficient is replaced 
                            # with the log of the total frame energy.
                            appendEnergy = False)
        mfcc_feature  = preprocessing.scale(features)
        print('ouput')
        return mfcc_feature


# #### Defining the GMM model into two one for female and one for male

# In[22]:


from sklearn.mixture import GaussianMixture
gmm_male=GaussianMixture(n_components=2, random_state=10)
gmm_female=GaussianMixture(n_components=2,random_state=10)


# In[23]:


male_samples=male_convert()
female_samples=female_convert()
Test_samples=input_convert()


# #### Fitting the samples of mfcc into the model

# In[24]:


gmm_male.fit(male_samples)
gmm_female.fit(female_samples)


# In[27]:


import pickle
filename = 'gmm_male.pkl'
with open(filename, 'wb') as file:
    pickle.dump(gmm_male, file)


# In[28]:


import pickle
filename = 'gmm_female.pkl'
with open(filename, 'wb') as file:
    pickle.dump(gmm_female, file)


# #### Making  a test variable which will store the sample to be predicted

# In[29]:


filen='gmm_male.pkl'
with open(filen, 'rb') as file:
    male_model = pickle.load(file)


# In[30]:


filen='gmm_female.pkl'
with open(filen, 'rb') as file:
    female_model = pickle.load(file)


# In[31]:


input_sample=Test_samples[92]

input_sample_reshaped=input_sample.reshape(1,-1)
selected=input_sample_reshaped[:,:6]


# ####  Predicting the sample

# In[32]:


male_likelihood=male_model.score_samples(selected)
print(male_likelihood)
female_likelihood=female_model.score_samples(selected)
print(female_likelihood)
predict_gender= "male" if male_likelihood > female_likelihood else "female"
print(f"predicted Gender of the audio file is:{predict_gender}")


# In[ ]:




