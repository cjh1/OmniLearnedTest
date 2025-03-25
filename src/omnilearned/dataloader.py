import torch
import torch.nn as nn
import h5py
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group, get_rank
import numpy as np
import random
import requests
import re, os
from urllib.parse import urljoin

def collate_point_cloud(batch):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    # Extract fields from batch
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]
    
    # Optional fields
    batch_c = [item["cond"] for item in batch if "cond" in item]
    batch_pid = [item["pid"] for item in batch if "pid" in item]

    batch_add_info = [item["add_info"] for item in batch if "add_info" in item]

    # Stack point clouds and labels
    point_clouds = torch.stack(batch_X)  # Shape: (B, N, F)
    labels = torch.stack(batch_y)        # Shape: (B, num_classes)

    # Determine valid particles (assuming last feature determines validity)
    valid_mask = (point_clouds[:, :, 2] != 0)  # Shape: (B, N)
    valid_counts = valid_mask.sum(dim=1)        # Number of valid particles per batch
    max_particles = valid_counts.max().item()   # M: max valid points across batch

    # Truncate point clouds to first `max_particles`
    truncated_X = point_clouds[:, :max_particles, :]  # Shape: (B, M, F)

    # Handle optional fields
    result = {"X": truncated_X, "y": labels}

    if batch_c:
        result["cond"] = torch.stack(batch_c)
    else:
        result["cond"] = None
    if batch_pid:
        result["pid"] = torch.stack(batch_pid)[:, :max_particles]
    else:
        result["pid"] = None
    if batch_add_info:
        result["add_info"] = torch.stack(batch_add_info)[:, :max_particles]
    else:
        result["add_info"] = None
        
    return result

def get_url(dataset_name,dataset_type,
            base_url = "https://portal.nersc.gov/cfs/m4567/"):
    
    url = f'{base_url}/{dataset_name}/{dataset_type}/'
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return url
    except requests.RequestException:
        return None
    
def download_h5_files(base_url, destination_folder):
    """
    Downloads all .h5 files from the specified directory URL.

    Args:
        base_url (str): The base URL of the directory containing the .h5 files.
        destination_folder (str): The local folder to save the downloaded files.
    """

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}")
        return


    file_links = re.findall(r'href="([^"]+\.h5)"', response.text)

    for file_name in file_links:
        file_url = urljoin(base_url, file_name)
        file_path = os.path.join(destination_folder, file_name)

        print(f"Downloading {file_url} to {file_path}")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_name}")

class HEPDataset(Dataset):
    def __init__(self, file_paths, 
                 use_pid = False, pid_idx = -1,
                 use_add = False,num_add = 4):
        """
        Args:
            file_paths (list): List of file paths.
            use_pid (bool): Flag to select if PID information is used during training
            use_add (bool): Flags to select if additional information besides kinematics are used
        """
        self.use_pid = use_pid
        self.use_add = use_add
        self.pid_idx = pid_idx
        self.num_add = num_add
        
        self.file_paths = file_paths
        self.file_indices = []  # [(file_index, sample_index), ...]
        self._file_cache = {}  # lazy cache for open h5py.File handles

        # Precompute indices for efficient access
        for file_idx, path in enumerate(self.file_paths):
            try:
                with h5py.File(path, 'r') as f:
                    num_samples = len(f['data'])
                    self.file_indices.extend([(file_idx, i) for i in range(num_samples)])
            except Exception as e:
                print(f"ERROR: File {path} is likely corrupted: {e}")
                
        random.shuffle(self.file_indices)  # Shuffle data entries globally

    def __len__(self):
        return len(self.file_indices)

    def _get_file(self, file_idx):
        # Get the file handle from cache; open it if it’s not already open.
        if file_idx not in self._file_cache:
            file_path = self.file_paths[file_idx]
            self._file_cache[file_idx] = h5py.File(file_path, 'r')
        return self._file_cache[file_idx]

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_indices[idx]
        f = self._get_file(file_idx)
        
        sample = {}

        sample['X'] = torch.tensor(f['data'][sample_idx], dtype=torch.float32)
        label = f['pid'][sample_idx]
        sample['y'] = torch.tensor(label, dtype=torch.int64)
        
        if 'global' in f:
            sample['cond'] = torch.tensor(f['global'][sample_idx], dtype=torch.float32)

            
        if self.use_pid:
            sample['pid'] = sample['X'][:,self.pid_idx].int()
            sample['X'] = torch.cat((sample['X'][:, :self.pid_idx], sample['X'][:, self.pid_idx+1:]), dim=1)
        if self.use_add:
            #Assume any additional info appears last
            sample['add_info'] = sample['X'][:,-self.num_add:]
            sample['X'] = sample['X'][:,:-self.num_add]
        return sample 

    def __del__(self):
        # Clean up: close all cached file handles.
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception as e:
                print(f"Error closing file: {e}")
        
def load_data(dataset_name,path,
              batch = 100,
              dataset_type = 'train',
              distributed = True,
              use_pid=False, pid_idx = 4,
              use_add = False,num_add = 4):
    
    supported_datasets = ['top', 'qg', 'pretrain', 'atlas','aspen', 'jetclass', 'jetclass2', 'h1']
    if dataset_name not in supported_datasets:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}.")

    if dataset_name == 'pretrain':
        names = ['atlas','aspen', 'jetclass',  'h1']
    else:
        names = [dataset_name]
        
    dataset_paths = [os.path.join(path,name,dataset_type) for name in names]
    
    file_list = []
    for iname, dataset_path in enumerate(dataset_paths):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        if not os.listdir(dataset_path):        
            print(f"Fetching download url for dataset {names[iname]}")
            url = get_url(names[iname],dataset_type)
            if url is None:
                raise ValueError(f"No download URL found for dataset '{dataset_name}'.")
            download_h5_files(url, dataset_path)
            
        file_list += [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

    data = HEPDataset(file_list,
                      use_pid=use_pid, pid_idx = pid_idx,
                      use_add = use_add,num_add = num_add)

    loader = DataLoader(data,batch_size=batch,
                        pin_memory=torch.cuda.is_available(),
                        #shuffle=False,
                        sampler=DistributedSampler(data, shuffle=dataset_type=='train') if distributed else None,
                        num_workers=16,
                        drop_last=True,
                        collate_fn=collate_point_cloud)
    return loader

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d","--dataset",default="top",help="Dataset name to download",)
    parser.add_argument("-f","--folder",default="./",help="Folder to save the dataset",)
    args = parser.parse_args()

    for tag in ['train','test','val']:           
        load_data(args.dataset,args.folder,dataset_type = tag, distributed=False)
