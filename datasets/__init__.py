
##datasets init
from .mnist import get_mnist_dataloader
from .cifar10 import get_cifar10_dataloader

# define the interface
__all__ = ["get_mnist_dataloader","get_cifar10_dataloader"]
