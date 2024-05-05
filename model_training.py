
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

# Print the length the training data
print('Training data size before split:', len(train_data))

# Define the ratio for the validation split
validation_ratio = 0.1

# Create a dictionary to hold speaker data
speaker_data = {}
for data in train_data:
    gender, speaker_id, _, _ = path2info(data['filename'])
    if speaker_id not in speaker_data:
        speaker_data[speaker_id] = {
            'gender': gender,
            'files': []
        }
    speaker_data[speaker_id]['files'].append(data)

# Separate speakers into training and validation sets
all_speakers = list(speaker_data.keys())
np.random.seed(0)  # for reproducibility
np.random.shuffle(all_speakers)
n_val_speakers = int(len(all_speakers) * validation_ratio)
val_speakers = set(all_speakers[:n_val_speakers])
train_speakers = set(all_speakers[n_val_speakers:])

# Gather the actual data for training and validation
X_val = [item for speaker in val_speakers for item in speaker_data[speaker]['files']]
X_train = [item for speaker in train_speakers for item in speaker_data[speaker]['files']]

# Function to count genders
def count_gender(dataset):
    count = {'man': 0, 'woman': 0}
    for data in dataset:
        _, _, _, _ = path2info(data['filename'])
        gender = path2info(data['filename'])[0]
        count[gender] += 1
    return count

# Counting genders in each set
gender_count_train = count_gender(X_train)
gender_count_val = count_gender(X_val)

# Print results
print("Training set size:", len(X_train), "Validation set size:", len(X_val))
print("Training set gender distribution:", gender_count_train)
print("Validation set gender distribution:", gender_count_val)

import numpy as np
from tqdm import tqdm

def get_features(data, dynamic=True):
    """
    Extracts features from the provided .npz data.

    Parameters:
        data (list): A list of dictionaries containing 'lmfcc', 'mspec', and 'targets'.
        dynamic (bool): If True, extracts dynamic features, otherwise extracts static features.

    Dynamic features:
        Stacks 7 MFCC or filterbank features symmetrically distributed around the current time step.
        For a time step n, it stacks the features at [n-3, n-2, n-1, n, n+1, n+2, n+3].
        At the start and end of each utterance, mirrors feature vectors to handle missing vectors.

    Returns:
        tuple: Tuple containing arrays for MFCC features, Melspectrogram features, and targets.
    """
    # Dimensions of feature vectors
    D_lmfcc = data[0]['lmfcc'].shape[1]
    D_mspec = data[0]['mspec'].shape[1]

    # Total number of frames across all samples
    N = sum(len(x['targets']) for x in data)

    # Initialize feature arrays
    if dynamic:
        mfcc_features = np.zeros((N, D_lmfcc * 7))
        mspec_features = np.zeros((N, D_mspec * 7))
    else:
        mfcc_features = np.zeros((N, D_lmfcc))
        mspec_features = np.zeros((N, D_mspec))

    targets = []
    k = 0

    # Process each entry in data
    for x in tqdm(data):
        n_frames = x['lmfcc'].shape[0]

        # Process each frame in the current entry
        for i in range(n_frames):
            if dynamic:
                # Use mirroring for boundary frames
                if i < 3 or i >= n_frames - 3:
                    padded_lmfcc = np.pad(x['lmfcc'], pad_width=((3, 3), (0, 0)), mode='reflect')
                    padded_mspec = np.pad(x['mspec'], pad_width=((3, 3), (0, 0)), mode='reflect')
                    mfcc_features[k, :] = np.hstack(padded_lmfcc[i:i + 7, :])
                    mspec_features[k, :] = np.hstack(padded_mspec[i:i + 7, :])
                else:
                    mfcc_features[k, :] = np.hstack(x['lmfcc'][i - 3:i + 4, :])
                    mspec_features[k, :] = np.hstack(x['mspec'][i - 3:i + 4, :])
            else:
                mfcc_features[k, :] = x['lmfcc'][i, :]
                mspec_features[k, :] = x['mspec'][i, :]

            k += 1

        # Append targets
        targets.extend(x['targets'])

    return mfcc_features, mspec_features, targets

# Dynamic features for train set
dyn_lmfcc_train_x, dyn_mspec_train_x, train_y = get_features(X_train,dynamic=True)
# Dynamic features for valisation set
dyn_lmfcc_val_x, dyn_mspec_val_x, val_y = get_features(X_val,dynamic=True)
# Dynamic features for test set
dyn_lmfcc_test_x, dyn_mspec_test_x, test_y = get_features(test_data,dynamic=True)

# non dynamic features
lmfcc_train_x, mspec_train_x, _ = get_features(X_train,dynamic=False)
lmfcc_val_x, mspec_val_x, _ = get_features(X_val,dynamic=False)
lmfcc_test_x, mspec_test_x, _ = get_features(test_data,dynamic=False)

# Starting with dynamic features

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the training data (calculate mean and variance of each feature)
scaler.fit(dyn_lmfcc_train_x)

# Transform the training data (apply normalization)
nor_dyn_lmfcc_train_x = scaler.transform(dyn_lmfcc_train_x)

# For validation or test sets, you would use the same scaler object:
nor_dyn_lmfcc_val_x = scaler.transform(dyn_lmfcc_val_x)
nor_dyn_lmfcc_test_x = scaler.transform(dyn_lmfcc_test_x)


# View the shapes
print(nor_dyn_lmfcc_train_x.shape)
print(nor_dyn_lmfcc_val_x.shape)
print(nor_dyn_lmfcc_test_x.shape)


# Moving to none dynamic features

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

# View the shapes
print(nor_lmfcc_train_x.shape)
print(nor_lmfcc_val_x.shape)
print(nor_lmfcc_test_x.shape)

# View target shapes
print(len(train_y))
print(len(val_y))
print(len(test_y))

# Convert feature arrays to 32 bits floating point format because of the hardware limitation in most GPUs
nor_dyn_lmfcc_train_x = nor_dyn_lmfcc_train_x.astype('float32')
nor_dyn_lmfcc_val_x = nor_dyn_lmfcc_val_x.astype('float32')
nor_dyn_lmfcc_test_x = nor_dyn_lmfcc_test_x.astype('float32')

nor_lmfcc_train_x = nor_lmfcc_train_x.astype('float32')
nor_lmfcc_val_x = nor_lmfcc_val_x.astype('float32')
nor_lmfcc_test_x = nor_lmfcc_test_x.astype('float32')

# Convert the target arrays into a one-hot encoding
import torch
import torch.nn.functional as F

output_dim = len(stateList)

train_y = F.one_hot(torch.tensor(train_y),num_classes=output_dim)
val_y = F.one_hot(torch.tensor(val_y),num_classes=output_dim)
test_y = F.one_hot(torch.tensor(test_y),num_classes=output_dim)

# This file contains boiler-plate code for defining and training a network in PyTorch.
# Please see PyTorch documentation and tutorials for more information
# e.g. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter



# define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # Define the output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)


        # define the layers of your network here
        # ...
    def forward(self, x):
        # define the foward computation from input to output
        # ...

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)

        return x

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# prepare/load the data into tensors
train_x = torch.tensor(nor_dyn_lmfcc_train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
val_x = torch.tensor(nor_dyn_lmfcc_val_x, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)
test_x = torch.tensor(nor_dyn_lmfcc_test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

input_size = 91
output_size = 61
# print(input_size, output_size)
hidden_sizes = [256, 256, 256, 256]

# instantiate the network and print the structure
net = Net()
print(net)
print('number of prameters:', count_parameters(net))

# define your loss criterion (see https://pytorch.org/docs/stable/nn.html#loss-functions)
criterion = nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.Adam(net.parameters())



batch_size = 64

# create the data loaders for training and validation sets
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# setup logging so that you can follow training using TensorBoard (see https://pytorch.org/docs/stable/tensorboard.html)
# writer = SummaryWriter()

# train the network
num_epochs = 100

for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()
        labels = torch.tensor(labels, dtype=torch.float32)  # Add this line
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # accumulate the training loss
        train_loss += loss.item()
        total_train+=batch_size
        _, predicted_train = torch.max(outputs, 1)
        _, targets = torch.max(labels, 1)
        correct_train += (predicted_train == targets).sum()
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    # calculate the validation loss
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        total_val = 0
        correct_val = 0
        for inputs, labels in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted_val = torch.max(outputs, 1)
            _, targets = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == targets).sum()
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)

    # print the epoch loss
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}')
    # writer.add_scalars('loss',{'train':train_loss,'val':val_loss},epoch)

# finally evaluate model on the test set here
net.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _, targets = torch.max(outputs, 1)
        total += batch_size
        # print(predicted.shape, labels.shape)
        correct += (predicted == targets).sum().item()

test_accuracy = correct / total
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# save the trained network
torch.save(net.state_dict(), 'trained-net.pt')

# finally evaluate model on the test set here
net.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _, target = torch.max(labels, 1)
        total += 256
        correct += (predicted == target).sum()

test_accuracy = correct / total
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


