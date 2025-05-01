from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
import random

class Dataset(ABC):
    """Abstract base class for datasets."""
    
    def __init__(self):
        self.data = None
        self.labels = None
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
        
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a sample and its label."""
        pass

class ArrayDataset(Dataset):
    """Dataset for numpy arrays."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Initialize dataset with numpy arrays.
        
        Args:
            X: Input data
            y: Labels (optional)
        """
        self.X = X
        self.y = y
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return a sample and its label."""
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], None

class DataLoader:
    """Data loader for batching and shuffling data."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False
    ):
        """
        Initialize data loader.
        
        Args:
            X: Input data
            y: Labels (optional)
            batch_size: Size of batches
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = ArrayDataset(X, y)
        self.batch_size = batch_size or len(X)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(self.dataset)))
        
    def __iter__(self):
        """Return iterator over batches."""
        if self.shuffle:
            random.shuffle(self.indices)
            
        batch = []
        for idx in self.indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self._collate_batch(batch)
                batch = []
                
        if batch and not self.drop_last:
            yield self._collate_batch(batch)
            
    def _collate_batch(self, batch: List[Tuple[np.ndarray, Optional[np.ndarray]]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Collate a batch of samples."""
        X_batch = np.stack([x for x, _ in batch])
        if batch[0][1] is not None:
            y_batch = np.stack([y for _, y in batch])
            return X_batch, y_batch
        return X_batch, None
        
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class Preprocessor:
    """Data preprocessing utilities."""
    
    @staticmethod
    def normalize(X: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize data to zero mean and unit variance.
        
        Args:
            X: Input data
            mean: Mean to use (if None, computed from data)
            std: Standard deviation to use (if None, computed from data)
            
        Returns:
            Normalized data, mean, and standard deviation
        """
        if mean is None:
            mean = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            
        return (X - mean) / std, mean, std
        
    @staticmethod
    def standardize(X: np.ndarray, min_val: Optional[np.ndarray] = None, max_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize data to [0, 1] range.
        
        Args:
            X: Input data
            min_val: Minimum value to use (if None, computed from data)
            max_val: Maximum value to use (if None, computed from data)
            
        Returns:
            Standardized data, minimum, and maximum
        """
        if min_val is None:
            min_val = np.min(X, axis=0)
        if max_val is None:
            max_val = np.max(X, axis=0)
            
        return (X - min_val) / (max_val - min_val), min_val, max_val
        
    @staticmethod
    def one_hot(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        """
        Convert labels to one-hot encoding.
        
        Args:
            y: Labels
            num_classes: Number of classes (if None, inferred from data)
            
        Returns:
            One-hot encoded labels
        """
        if num_classes is None:
            num_classes = np.max(y) + 1
        return np.eye(num_classes)[y]
        
    @staticmethod
    def split_data(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.8,
        val_size: float = 0.1,
        shuffle: bool = True
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Input data
            y: Labels
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Training, validation, and test sets
        """
        n_samples = len(X)
        indices = list(range(n_samples))
        
        if shuffle:
            random.shuffle(indices)
            
        train_end = int(n_samples * train_size)
        val_end = train_end + int(n_samples * val_size)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return (
            (X[train_indices], y[train_indices]),
            (X[val_indices], y[val_indices]),
            (X[test_indices], y[test_indices])
        ) 