from jax import tree_map
import jax.numpy as jnp
from torch.utils import data
from torchvision.datasets import FashionMNIST, MNIST
import numpy as np

#https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html#data-loading-with-pytorch

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
  
class TransformImage:
  def __call__(self, image):
    return np.array(image, dtype=jnp.float32).reshape(28, 28, 1)/255.0
 

def get_dataset(name: str):
    if name == "fashion-mnist":
        fashion_mnist_train_ds = FashionMNIST(
            './data/', 
            download=True,
            train=True,
            transform=TransformImage(),
        )
        fashion_mnist_test_ds = FashionMNIST(
            './data/', 
            download=True, 
            train=False,
        )
        return fashion_mnist_train_ds, fashion_mnist_test_ds
    elif name == "mnist":
        mnist_train_ds = MNIST(
            './data/', 
            download=True,
            train=True,
            transform=TransformImage(),
        )
        mnist_test_ds = MNIST(
            './data/', 
            download=True, 
            train=False,
        )
        return mnist_train_ds, mnist_test_ds
    else:
      raise NotImplementedError
    

def get_dataloader(dataset: data.Dataset, batch_size: int, shuffle: bool = True) -> data.DataLoader:
    dataloader = NumpyLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader