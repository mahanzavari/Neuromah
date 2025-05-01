from typing import Dict, List, Optional, Any, Callable
import numpy as np
import cupy as cp
from ..layers.core.base import BaseLayer
from . import Graph, Node, ExecutionMode, registry

def layer_to_node(layer: BaseLayer) -> Node:
    """Convert a layer to a graph node."""
    node = Node(
        name=layer.__class__.__name__,
        op=layer.__class__.__name__,
        inputs=[],
        outputs=[],
        attributes={
            'parameters': layer.get_parameters() if hasattr(layer, 'get_parameters') else {},
            'use_gpu': layer.use_gpu if hasattr(layer, 'use_gpu') else False
        }
    )
    return node

def model_to_graph(model: 'Model') -> Graph:
    """Convert a model to a computational graph."""
    graph = Graph(name=model.__class__.__name__)
    
    # Create nodes for each layer
    layer_nodes = {}
    for layer in model.layers:
        node = layer_to_node(layer)
        layer_nodes[layer] = node
        graph.add_node(node)
    
    # Connect nodes based on layer connections
    for i, layer in enumerate(model.layers):
        if i > 0:
            prev_layer = model.layers[i-1]
            layer_nodes[prev_layer].outputs.append(layer_nodes[layer])
            layer_nodes[layer].inputs.append(layer_nodes[prev_layer])
    
    # Set input and output nodes
    if model.layers:
        graph.input_nodes = [layer_nodes[model.layers[0]]]
        graph.output_nodes = [layer_nodes[model.layers[-1]]]
    
    return graph

def register_layer_ops():
    """Register layer operations in the graph registry."""
    
    def layer_forward(layer: BaseLayer, inputs: np.ndarray) -> np.ndarray:
        """Forward pass operation for a layer."""
        layer.forward(inputs, training=True)
        return layer.output
    
    def layer_backward(layer: BaseLayer, dvalues: np.ndarray) -> np.ndarray:
        """Backward pass operation for a layer."""
        layer.backward(dvalues)
        return layer.dinputs
    
    # Register forward operations
    registry.register_op(
        'Dense',
        layer_forward,
        lambda inputs: (inputs.shape[0], layer.units),
        lambda inputs: inputs.dtype
    )
    
    registry.register_op(
        'Conv2D',
        layer_forward,
        lambda inputs: (
            inputs.shape[0],
            layer.output_channels,
            (inputs.shape[2] + 2 * layer.padding - layer.kernel_size) // layer.stride + 1,
            (inputs.shape[3] + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        ),
        lambda inputs: inputs.dtype
    )
    
    registry.register_op(
        'Pooling',
        layer_forward,
        lambda inputs: (
            inputs.shape[0],
            inputs.shape[1],
            (inputs.shape[2] + 2 * layer.padding - layer.pool_size) // layer.stride + 1,
            (inputs.shape[3] + 2 * layer.padding - layer.pool_size) // layer.stride + 1
        ),
        lambda inputs: inputs.dtype
    )
    
    registry.register_op(
        'Dropout',
        layer_forward,
        lambda inputs: inputs.shape,
        lambda inputs: inputs.dtype
    )
    
    # Register backward operations
    registry.register_op(
        'Dense_backward',
        layer_backward,
        lambda dvalues: dvalues.shape,
        lambda dvalues: dvalues.dtype
    )
    
    registry.register_op(
        'Conv2D_backward',
        layer_backward,
        lambda dvalues: dvalues.shape,
        lambda dvalues: dvalues.dtype
    )
    
    registry.register_op(
        'Pooling_backward',
        layer_backward,
        lambda dvalues: dvalues.shape,
        lambda dvalues: dvalues.dtype
    )
    
    registry.register_op(
        'Dropout_backward',
        layer_backward,
        lambda dvalues: dvalues.shape,
        lambda dvalues: dvalues.dtype
    )

class GraphModel:
    """Wrapper class for models with graph execution support."""
    
    def __init__(self, model: 'Model'):
        self.model = model
        self.graph = model_to_graph(model)
        self.execution_mode = ExecutionMode.EAGER
    
    def compile(self, mode: ExecutionMode = ExecutionMode.GRAPH) -> None:
        """Compile the model for execution in the specified mode."""
        self.execution_mode = mode
        self.graph.compile(mode)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass of the model."""
        if self.execution_mode == ExecutionMode.EAGER:
            return self.model.forward(inputs)
        else:
            return self.graph.execute({'input': inputs})['output']
    
    def backward(self, dvalues: np.ndarray) -> None:
        """Backward pass of the model."""
        if self.execution_mode == ExecutionMode.EAGER:
            self.model.backward(dvalues)
        else:
            self.graph.execute({'dvalues': dvalues})
    
    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Training step of the model."""
        if self.execution_mode == ExecutionMode.EAGER:
            return self.model.train_step(inputs, targets)
        else:
            outputs = self.graph.execute({
                'input': inputs,
                'target': targets
            })
            return {
                'loss': outputs['loss'],
                'accuracy': outputs['accuracy']
            }
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Prediction step of the model."""
        if self.execution_mode == ExecutionMode.EAGER:
            return self.model.predict(inputs)
        else:
            return self.graph.execute({'input': inputs})['output']

# Register layer operations when the module is imported
register_layer_ops() 