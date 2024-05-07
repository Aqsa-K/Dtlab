#!/usr/bin/env python
# coding: utf-8

# # DT2119 Lab3: Model

# In[1]:


import os
import json
import io
import numpy as np
# from google.colab import files
# uploaded = files.upload()

# Assuming the filename of your key is 'service-account-file.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'dt2119-lab3-422421-d62072692aa2.json'

from google.cloud import storage

# This creates a client that uses the specified service account credentials
client = storage.Client()
# Now you can interact with Google Cloud Storage using this client

# Lists all the buckets
buckets = list(client.list_buckets())
print(buckets)


def read_from_storage(bucket_name, file_name):
    """Reads content from a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_name (str): The name of the file to read from the bucket.

    Returns:
        str: The content of the file.
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(file_name)

    # Download the content as a byte string
    content = blob.download_as_bytes()

    # Convert to string if necessary (assuming the file is text-based)
    return content.decode('utf-8')

# Example usage:
# content = read_from_storage('experimentresults', 'new-file.txt')
# print(content)

def write_to_storage(bucket_name, file_name, data):
    """Writes data to a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_name (str): The name of the file to write in the bucket.
        data (str): The data to write to the file.

    Returns:
        None
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(file_name)

    # Upload the data
    blob.upload_from_string(data)


# Example usage:
# write_to_storage('experimentresults', 'new-file-1.txt', 'Hello, World!')


def create_and_write_file(file_name, text_string):
    """
    Creates a file and writes a specified text string to it.

    Args:
        file_name (str): The name of the file to create.
        text_string (str): The text string to write to the file.

    Returns:
        None
    """
    # Open the file in write mode ('w'). If the file doesn't exist, it will be created.
    # If the file exists, it will be overwritten.
    with open(file_name, 'w') as file:
        file.write(text_string)

# Example usage:
# create_and_write_file('example.txt', 'Hello, this is a sample text.')


def write_json_to_gcs(bucket_name, destination_blob_name, data):
    """
    Writes JSON data to a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        destination_blob_name (str): The path within the bucket to save the file.
        data (str): JSON string to write to the file.
    """
    # Create a temporary file
    temp_file = "/tmp/tempfile.json"
    with open(temp_file, "w") as file:
        file.write(data)

    # Initialize the Google Cloud Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(temp_file)

    # Optionally, remove the temporary file if not needed
    os.remove(temp_file)


def read_json_from_gcs(bucket_name, source_blob_name):
    """
    Reads a JSON file from Google Cloud Storage and parses the JSON content.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the blob (file) in the GCS bucket.

    Returns:
        dict: The parsed JSON data as a dictionary.
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    # Download the content as a string
    json_data = blob.download_as_string()

    # Parse the JSON string into a Python dictionary
    data = json.loads(json_data)

    return data


# import json
#
# data = {
#     "name": "Example",
#     "age": 30,
#     "city": "New York"
# }
#
# # Convert the data to JSON format
# json_data = json.dumps(data)
#
# # Example usage:
# write_json_to_gcs('experimentresults', 'examplefile.json', json_data)
#
# # Example usage:
# data = read_json_from_gcs('experimentresults', 'examplefile.json')
# print(data)

import tempfile
def load_npz_from_gcs(bucket_name, blob_name):
    """Load a .npz file from Google Cloud Storage.

    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        blob_name (str): Name of the .npz file (blob) in the bucket.

    Returns:
        dict: A dictionary containing arrays loaded from the .npz file.
    """
    # Create a client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(blob_name)

    # Download the blob into memory
    content = blob.download_as_string()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Download the blob to the temporary file
        blob.download_to_filename(temp_file.name)

        # Load .npz file
        data = np.load(temp_file.name, allow_pickle=True)

    return data

bucket_name = 'dt2119-project'
train_data = load_npz_from_gcs(bucket_name, 'data/train_data.npz')['train_data']
test_data = load_npz_from_gcs(bucket_name, 'data/test_data.npz')['test_data']
stateList = load_npz_from_gcs(bucket_name, 'data/statelist.npz')['arr_0']



import numpy as np
from tqdm import tqdm
import os
#from pysndfile import sndio
import soundfile as sf

def path2info(path):
    """
    path2info: parses paths in the TIDIGIT format and extracts information
               about the speaker and the utterance

    Example:
    path2info('tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav')
    """
    rest, filename = os.path.split(path)
    rest, speakerID = os.path.split(rest)
    rest, gender = os.path.split(rest)
    digits = filename[:-5]
    repetition = filename[-5]
    return gender, speakerID, digits, repetition

def get_features(data,dynamic=True):
    '''
    Get fetures from the .npz data, if dynamic=True, get the dynamic features
    Dynamic features: stack 7 MFCC or filterbank features symmetrically distributed around the current time step. 
    At time n, stack the features at times [n − 3, n − 2, n − 1, n, n + 1, n + 2, n + 3]). 
    At the beginning and end of each utterance, use mirrored feature vectors in place of the missing vectors.
    '''
    D_lmfcc = data[0]['lmfcc'].shape[1]
    D_mspec = data[0]['mspec'].shape[1]
    N = sum([len(x['targets']) for x in data])
    
    if dynamic:
        mfcc_features = np.zeros((N,D_lmfcc*7))
        mspec_features = np.zeros((N,D_mspec*7))
    else:
        mfcc_features = np.zeros((N,D_lmfcc))
        mspec_features = np.zeros((N,D_mspec))
    
    targets = []
    
    k = 0
    for x in tqdm(data): 
        n_frames, dim = x['lmfcc'].shape

        ## for each timestep
        for i in range(n_frames):
            if dynamic:
                if i< 3 or i >= n_frames-3:
                    mfcc_features[k,:] = np.hstack(np.pad(x['lmfcc'], pad_width=((3, 3), (0, 0)), mode='reflect')[i:i+7,:])
                    mspec_features[k,:] = np.hstack(np.pad(x['mspec'], pad_width=((3, 3), (0, 0)), mode='reflect')[i:i+7,:])
                else:
                    mfcc_features[k,:] = np.hstack(x['lmfcc'][i-3:i+4,:])
                    mspec_features[k,:] = np.hstack(x['mspec'][i-3:i+4,:])
            else:
                mfcc_features[k,:] = x['lmfcc'][i,:]
                mspec_features[k,:] = x['mspec'][i,:]
            
            k +=1
   
        targets = targets + x['targets']
    
    return mfcc_features, mspec_features, targets


# Load training data
# train_data = np.load('train_data.npz', allow_pickle=True)['train_data']

N = len(train_data)
n_val = round(N*0.1)

indexes = np.random.permutation(N)
X = np.take(train_data,indexes)
X_val = X[:n_val]
X_train = X[n_val:]

N = len(X) + len(X_val)
print(len(X)/N,len(X_val)/N)
# count men and women
## Trainingset
N_X_women = sum(1 for data in X if data['filename'].split("/")[-3] == 'woman')
N_X_man = len(X)-N_X_women
## Validationset
N_X_val_women = sum(1 for data in X_val if data['filename'].split("/")[-3] == 'woman')
N_X_val_man = len(X_val)-N_X_val_women


print("Trainign set: Women->",N_X_women,", Men->",N_X_man)
print("Validation set: Women->",N_X_val_women,", Men->",N_X_val_man)

# Load test data
# test_data = np.load('test_data.npz', allow_pickle=True)['test_data']

# dynamic features
lmfcc_train_x, mspec_train_x, train_y = get_features(X_train,dynamic=True)
lmfcc_val_x, mspec_val_x, val_y = get_features(X_val,dynamic=True)
lmfcc_test_x, mspec_test_x, test_y = get_features(test_data,dynamic=True)

lmfcc_train_x = mspec_train_x
lmfcc_val_x = mspec_val_x
lmfcc_test_x = mspec_test_x

# In[3]:


# Normalizing lmfcc non dynamic features
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the training data (calculate mean and variance of each feature)
scaler.fit(lmfcc_train_x)

# Transform the training data (apply normalization)
nor_lmfcc_train_x = scaler.transform(lmfcc_train_x)

# For validation or test sets, you would use the same scaler object:
nor_lmfcc_val_x = scaler.transform(lmfcc_val_x)
nor_lmfcc_test_x = scaler.transform(lmfcc_test_x)


# In[4]:


nor_lmfcc_val_x.shape


# In[ ]:


# Convert feature arrays to 32 bits floating point format because of the hardware limitation in most GPUs
nor_lmfcc_train_x = nor_lmfcc_train_x.astype('float32')
nor_lmfcc_val_x = nor_lmfcc_val_x.astype('float32')
nor_lmfcc_test_x = nor_lmfcc_test_x.astype('float32')

# Convert the target arrays into a one-hot encoding
import torch
import torch.nn.functional as F
# from keras.utils import np_utils

# Load stateList data
# stateList = np.load('./statelist.npz', allow_pickle=True)["stateList"]
output_dim = len(stateList)
print(stateList)

# targets_train = np_utils.to_categorical(train_y, output_dim)
# targets_val = np_utils.to_categorical(val_y, output_dim)
# targets_test = np_utils.to_categorical(test_y, output_dim)

train_y = F.one_hot(torch.tensor(train_y),num_classes=output_dim)
val_y = F.one_hot(torch.tensor(val_y),num_classes=output_dim)
test_y = F.one_hot(torch.tensor(test_y),num_classes=output_dim)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation, as CrossEntropyLoss will apply Softmax
        return x


# ### Step 2: Define Input and Output Sizes
# Given the information:

# In[20]:


print("Input feature vector size:", nor_lmfcc_train_x.shape[1])
print("Number of output classes:", train_y.shape[1])


# ### Step 3: Choose the Output Activation Function
# 
# Since this is a classification problem, and our output is a probability distribution over classes, we chose `nn.Softmax()` for the output layer.
# 
# #### Alternatives:
# 
# - `nn.Sigmoid()` is used for binary classification.
# - `nn.Softmax()` is typically used for multi-class classification.
# - `nn.LogSoftmax()` can also be used for multi-class classification, but it requires `nn.NLLLoss()` as the loss criterion.
# 
# ### Step 4: Choose the Intermediate Activation Functions
# 
# For the intermediate layers, we chose `nn.ReLU()` which is a common choice for hidden layers in neural networks. Other options could include:
# 
# - `nn.Sigmoid()` - a smoother function but prone to vanishing gradient issues.
# - `nn.Tanh()` - another smoother function but with a similar downside to `nn.Sigmoid()`.
# 
# ### Step 5: Choose the Loss Criterion and Optimizer
# 
# For multi-class classification, the appropriate loss criterion is `nn.CrossEntropyLoss()`. For the optimizer, we chose Adam (`torch.optim.Adam()`).
# 

# In[21]:


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
    
# instantiate the network
net = Net(input_size=nor_lmfcc_train_x.shape[1], output_size=train_y.shape[1])
print(net)
print('number of parameters:', count_parameters(net))

# define the loss criterion
criterion = nn.CrossEntropyLoss()

# define the optimizer
optimizer = optim.Adam(net.parameters())


# ### Step 6: Load Data
# We use the previous data treated to create the DataLoader objects for training and validation data. 

# In[22]:


# Create input and output tensors
X_tensor = torch.tensor(nor_lmfcc_train_x, dtype=torch.float32)
Y_tensor = torch.tensor(train_y, dtype=torch.long)
X_val_tensor = torch.tensor(nor_lmfcc_val_x, dtype=torch.float32)
Y_val_tensor = torch.tensor(val_y, dtype=torch.long)

# Create the dataset and dataloader
train_dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


# ### Step 7: Train the Network

# In[23]:


# from torch.utils.tensorboard import SummaryWriter


# setup logging
# writer = SummaryWriter()

# Initialize lists to track losses
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

num_epochs = 15
for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()

    net.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == torch.argmax(labels, dim=1)).sum().item()

    # Calculate average losses
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # Calculate accuracies
    train_accuracy = 100 * correct_train / total_train
    val_accuracy = 100 * correct_val / total_val

    # Append the epoch losses and accuracies to the lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch}: train_loss={train_loss}, train_accuracy={train_accuracy}%, val_loss={val_loss}, val_accuracy={val_accuracy}%')
    # writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)

# save the trained network
torch.save(net.state_dict(), 'trained-net.pt')

def save_pt_to_gcs(bucket_name, blob_name, local_file_path):
    """Save a .pt file to Google Cloud Storage.

    Args:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the blob (file) in the bucket.
        local_file_path (str): Local path to the .pt file to be saved.
    """
    # Create a client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Create a blob
    blob = bucket.blob(blob_name)

    # Upload the local file to GCS
    blob.upload_from_filename(local_file_path)

# Example usage:
bucket_name = 'dt2119-project'
blob_name = 'trained-net.pt'
local_file_path = 'trained-net.pt'
save_pt_to_gcs(bucket_name, blob_name, local_file_path)

# Convert lists to numpy arrays
train_losses_np = np.array(train_losses)
val_losses_np = np.array(val_losses)
train_accuracies_np = np.array(train_accuracies)
val_accuracies_np = np.array(val_accuracies)

# Save numpy arrays to files
np.save('train_losses.npy', train_losses_np)
np.save('val_losses.npy', val_losses_np)
np.save('train_accuracies.npy', train_accuracies_np)
np.save('val_accuracies.npy', val_accuracies_np)


# Define the name of your GCS bucket and file paths
blob_name_train_losses = 'train_losses.npy'
blob_name_val_losses = 'val_losses.npy'
blob_name_train_accuracies = 'train_accuracies.npy'
blob_name_val_accuracies = 'val_accuracies.npy'

save_pt_to_gcs(bucket_name, blob_name_train_losses, blob_name_train_losses)
save_pt_to_gcs(bucket_name, blob_name_val_losses, blob_name_val_losses)
save_pt_to_gcs(bucket_name, blob_name_train_accuracies, blob_name_train_accuracies)
save_pt_to_gcs(bucket_name, blob_name_val_accuracies, blob_name_val_accuracies)


# In[25]:


# # Plotting the losses
# import matplotlib.pyplot as plt

# # Plot the losses
# plt.figure(figsize=(10, 5))
# plt.plot(range(num_epochs), train_losses, label='Training Loss')
# plt.plot(range(num_epochs), val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss over Epochs')
# plt.grid(True)
# plt.show()

