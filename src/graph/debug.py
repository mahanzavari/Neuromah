from typing import Dict, List, Optional, Any
import numpy as np
import traceback
from . import Graph, Node, ExecutionMode

class GraphDebugger:
    """Provides debugging and introspection tools for computational graphs."""
    
    def __init__(self):
        self._error_history = []
        self._warnings = []
    
    def check_graph(self, graph: Graph) -> List[str]:
        """Check the graph for potential issues."""
        issues = []
        
        # Check for disconnected nodes
        for node in graph.nodes.values():
            if not node.inputs and node.op != 'input':
                issues.append(f"Node {node.name} has no inputs")
            if not node.outputs and node.op != 'output':
                issues.append(f"Node {node.name} has no outputs")
        
        # Check for shape mismatches
        for node in graph.nodes.values():
            if node.shape is not None:
                for input_node in node.inputs:
                    if input_node.shape is not None and input_node.shape != node.shape:
                        issues.append(
                            f"Shape mismatch: {input_node.name} ({input_node.shape}) -> "
                            f"{node.name} ({node.shape})"
                        )
        
        # Check for dtype mismatches
        for node in graph.nodes.values():
            if node.dtype is not None:
                for input_node in node.inputs:
                    if input_node.dtype is not None and input_node.dtype != node.dtype:
                        issues.append(
                            f"Dtype mismatch: {input_node.name} ({input_node.dtype}) -> "
                            f"{node.name} ({node.dtype})"
                        )
        
        return issues
    
    def handle_error(self, error: Exception, graph: Graph) -> str:
        """Handle and format error messages for graph execution."""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'graph_name': graph.name,
            'execution_mode': graph.execution_mode.name
        }
        self._error_history.append(error_info)
        
        # Format error message
        message = f"Error in graph '{graph.name}' ({graph.execution_mode.name} mode):\n"
        message += f"{error_info['type']}: {error_info['message']}\n"
        
        # Add context-specific information
        if isinstance(error, ValueError):
            message += self._handle_value_error(error, graph)
        elif isinstance(error, RuntimeError):
            message += self._handle_runtime_error(error, graph)
        
        return message
    
    def _handle_value_error(self, error: ValueError, graph: Graph) -> str:
        """Handle ValueError with additional context."""
        message = "\nCommon causes:\n"
        message += "- Shape mismatches in operations\n"
        message += "- Invalid parameter values\n"
        message += "- Unsupported operation combinations\n"
        return message
    
    def _handle_runtime_error(self, error: RuntimeError, graph: Graph) -> str:
        """Handle RuntimeError with additional context."""
        message = "\nCommon causes:\n"
        message += "- Memory allocation failures\n"
        message += "- Device synchronization issues\n"
        message += "- Invalid operation sequences\n"
        return message
    
    def get_graph_summary(self, graph: Graph) -> str:
        """Generate a summary of the graph structure."""
        summary = f"Graph '{graph.name}' Summary:\n"
        summary += f"Execution Mode: {graph.execution_mode.name}\n"
        summary += f"Number of Nodes: {len(graph.nodes)}\n"
        summary += f"Input Nodes: {len(graph.input_nodes)}\n"
        summary += f"Output Nodes: {len(graph.output_nodes)}\n"
        
        # Add node details
        summary += "\nNode Details:\n"
        for node in graph.nodes.values():
            summary += f"- {node.name} ({node.op}):\n"
            summary += f"  Inputs: {[n.name for n in node.inputs]}\n"
            summary += f"  Outputs: {[n.name for n in node.outputs]}\n"
            if node.shape:
                summary += f"  Shape: {node.shape}\n"
            if node.dtype:
                summary += f"  Dtype: {node.dtype}\n"
            if node.device:
                summary += f"  Device: {node.device}\n"
        
        return summary
    
    def get_performance_summary(self, graph: Graph) -> str:
        """Generate a summary of graph performance."""
        # TODO: Implement performance summary
        return "Performance summary not available yet."
    
    def get_memory_summary(self, graph: Graph) -> str:
        """Generate a summary of memory usage."""
        # TODO: Implement memory summary
        return "Memory summary not available yet."

def enable_debug_mode() -> None:
    """Enable debug mode for graph execution."""
    # TODO: Implement debug mode
    pass

def disable_debug_mode() -> None:
    """Disable debug mode for graph execution."""
    # TODO: Implement debug mode
    pass

# Global debugger instance
debugger = GraphDebugger() 