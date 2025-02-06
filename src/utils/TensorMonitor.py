import numpy as np
import json
import os
import time
import inspect

class TensorMonitor:
    def __init__(self, log_dir='tensor_logs'):
        """
        Initializes the TensorMonitor for logging training data at the end of each epoch.
        Logs metadata and (epoch-wise) parameter histograms.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")  # Unique run ID based on timestamp
        self.log_file_path = os.path.join(self.log_dir, f'run-{self.run_id}.json')
        self.log_data = {
            'metadata': {
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'layers': [],  # Filled in start_run()
                'optimizer': '',
                'loss_function': '',
                'accuracy_metric': ''
            },
            'epochs': []  # List to store epoch-wise data
        }
        self.epoch_data = None  # Temporary storage for current epoch's data
        self.epoch_count = 0

    def start_run(self, model):
        """
        Initializes the log data with model architecture and settings.
        This extracts layer information only for those layers having attributes.
        """
        layer_configs = []
        for layer in model.layers:
            # Only log layers that have weights (or similar parameters)
            if hasattr(layer, 'weights'):
                config = {}
                for key, value in layer.__dict__.items():
                    # Only store simple types or convert arrays to lists
                    if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
                        config[key] = value
                    elif isinstance(value, np.ndarray):
                        config[key] = value.tolist()
                    elif not inspect.ismodule(value) and not inspect.isfunction(value) and not inspect.ismethod(value):
                        config[key] = str(value)
                layer_configs.append({'name': layer.__class__.__name__, 'config': config})
        self.log_data['metadata']['layers'] = layer_configs

        if model.optimizer:
            optimizer_config = {}
            if hasattr(model.optimizer, '__dict__'):
                for key, value in model.optimizer.__dict__.items():
                    if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
                        optimizer_config[key] = value
                    elif isinstance(value, np.ndarray):
                        optimizer_config[key] = value.tolist()
                    elif not inspect.ismodule(value) and not inspect.isfunction(value) and not inspect.ismethod(value):
                        optimizer_config[key] = str(value)
            self.log_data['metadata']['optimizer'] = model.optimizer.__class__.__name__ + " " + str(optimizer_config)

        if model.loss:
            self.log_data['metadata']['loss_function'] = model.loss.__class__.__name__
        if model.accuracy:
            self.log_data['metadata']['accuracy_metric'] = model.accuracy.__class__.__name__

    def start_epoch(self, epoch_num):
        """
        Starts logging data for a new epoch.
        """
        self.epoch_count = epoch_num
        self.epoch_data = {
            'epoch': epoch_num,
            'time_start': time.time(),
            'metrics': {},      # Scalars like loss/accuracy for the epoch
            'parameters': {}    # Histograms for parameters, logged per layer
        }

    def log_scalar(self, tag, value):
        """
        Logs a scalar value (e.g., loss, accuracy) for the current epoch.
        """
        if self.epoch_data is not None:
            if 'scalars' not in self.epoch_data['metrics']:
                self.epoch_data['metrics']['scalars'] = {}
            self.epoch_data['metrics']['scalars'][tag] = value

    def log_histogram(self, tag, values, bins=10):
        """
        Logs a histogram of values for the current epoch.
        """
        if self.epoch_data is not None:
            if 'histograms' not in self.epoch_data['parameters']:
                self.epoch_data['parameters']['histograms'] = {}
            hist, bin_edges = np.histogram(values.flatten(), bins=bins)
            self.epoch_data['parameters']['histograms'][tag] = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }

    def log_layer_parameters(self, layer):
        """
        If the layer has parameters (weights), logs the histograms for:
          - weights
          - biases (if available)
          - weight gradients (dweights) if available
          - bias gradients (dbiases) if available
        """
        if hasattr(layer, 'weights'):
            self.log_histogram(f'{layer.__class__.__name__}/weights', layer.weights)
            if hasattr(layer, 'weight_gradients'):
                self.log_histogram(f'{layer.__class__.__name__}/dweights', layer.weight_gradients)
            if hasattr(layer, 'biases'):
                self.log_histogram(f'{layer.__class__.__name__}/biases', layer.biases)
                if hasattr(layer, 'bias_gradients'):
                    self.log_histogram(f'{layer.__class__.__name__}/dbiases', layer.bias_gradients)

    def end_epoch(self):
        """
        Finalizes logging for the current epoch and saves epoch data.
        """
        if self.epoch_data is not None:
            self.epoch_data['time_end'] = time.time()
            self.epoch_data['time_elapsed'] = self.epoch_data['time_end'] - self.epoch_data['time_start']
            self.log_data['epochs'].append(self.epoch_data)
            self.epoch_data = None  # Reset for next epoch

    def save_logs(self):
        """
        Saves all collected log data to a JSON file.
        """
        self.log_data['metadata']['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file_path, 'w') as f:
            json.dump(self.log_data, f, indent=4, default=str)
        print(f"TensorMonitor logs saved to: {self.log_file_path}")
