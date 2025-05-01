import unittest
import numpy as np
import cupy as cp
from src.graph import Graph, Node, ExecutionMode
from src.graph.debug import GraphDebugger
from src.layers import Dense, ReLU, Softmax
from src.optimizers import Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy

class TestGraphIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Generate synthetic data
        self.n_samples = 1000
        self.n_features = 20
        self.n_classes = 3
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, self.n_classes, size=self.n_samples)
        
    def test_graph_creation(self):
        """Test basic graph creation and structure."""
        # Create a simple graph
        graph = Graph(name="test_graph")
        
        # Add input node
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        graph.add_node(input_node)
        
        # Add dense layer node
        dense_node = Node(name="dense", op="dense", shape=(64,))
        graph.add_node(dense_node)
        graph.add_edge(input_node, dense_node)
        
        # Add activation node
        relu_node = Node(name="relu", op="relu", shape=(64,))
        graph.add_node(relu_node)
        graph.add_edge(dense_node, relu_node)
        
        # Add output node
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        graph.add_node(output_node)
        graph.add_edge(relu_node, output_node)
        
        # Verify graph structure
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 3)
        self.assertEqual(graph.input_nodes, [input_node])
        self.assertEqual(graph.output_nodes, [output_node])
        
    def test_graph_execution(self):
        """Test graph execution in different modes."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense_node = Node(name="dense", op="dense", shape=(64,))
        relu_node = Node(name="relu", op="relu", shape=(64,))
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_node(relu_node)
        graph.add_node(output_node)
        
        graph.add_edge(input_node, dense_node)
        graph.add_edge(dense_node, relu_node)
        graph.add_edge(relu_node, output_node)
        
        # Test CPU execution
        graph.execution_mode = ExecutionMode.CPU
        input_data = np.random.randn(32, self.n_features)
        output = graph.execute(input_data)
        self.assertEqual(output.shape, (32, self.n_classes))
        
        # Test CUDA execution if available
        if cp.cuda.is_available():
            graph.execution_mode = ExecutionMode.CUDA
            input_data = cp.asarray(input_data)
            output = graph.execute(input_data)
            self.assertTrue(isinstance(output, cp.ndarray))
            self.assertEqual(output.shape, (32, self.n_classes))
            
    def test_graph_debugging(self):
        """Test graph debugging functionality."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense_node = Node(name="dense", op="dense", shape=(64,))
        relu_node = Node(name="relu", op="relu", shape=(64,))
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_node(relu_node)
        graph.add_node(output_node)
        
        graph.add_edge(input_node, dense_node)
        graph.add_edge(dense_node, relu_node)
        graph.add_edge(relu_node, output_node)
        
        # Create debugger
        debugger = GraphDebugger()
        
        # Check graph for issues
        issues = debugger.check_graph(graph)
        self.assertEqual(len(issues), 0)
        
        # Get graph summary
        summary = debugger.get_graph_summary(graph)
        self.assertIn("Graph 'test_graph' Summary", summary)
        self.assertIn("Number of Nodes: 4", summary)
        
    def test_graph_optimization(self):
        """Test graph optimization."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense1_node = Node(name="dense1", op="dense", shape=(64,))
        dense2_node = Node(name="dense2", op="dense", shape=(32,))
        relu1_node = Node(name="relu1", op="relu", shape=(64,))
        relu2_node = Node(name="relu2", op="relu", shape=(32,))
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense1_node)
        graph.add_node(dense2_node)
        graph.add_node(relu1_node)
        graph.add_node(relu2_node)
        graph.add_node(output_node)
        
        graph.add_edge(input_node, dense1_node)
        graph.add_edge(dense1_node, relu1_node)
        graph.add_edge(relu1_node, dense2_node)
        graph.add_edge(dense2_node, relu2_node)
        graph.add_edge(relu2_node, output_node)
        
        # Optimize graph
        graph.optimize()
        
        # Verify optimization
        self.assertTrue(graph.is_optimized)
        
    def test_graph_serialization(self):
        """Test graph serialization and deserialization."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense_node = Node(name="dense", op="dense", shape=(64,))
        relu_node = Node(name="relu", op="relu", shape=(64,))
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_node(relu_node)
        graph.add_node(output_node)
        
        graph.add_edge(input_node, dense_node)
        graph.add_edge(dense_node, relu_node)
        graph.add_edge(relu_node, output_node)
        
        # Serialize graph
        serialized = graph.serialize()
        
        # Deserialize graph
        new_graph = Graph.deserialize(serialized)
        
        # Verify deserialization
        self.assertEqual(len(new_graph.nodes), len(graph.nodes))
        self.assertEqual(len(new_graph.edges), len(graph.edges))
        self.assertEqual(new_graph.name, graph.name)
        
    def test_graph_error_handling(self):
        """Test graph error handling."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes with shape mismatch
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense_node = Node(name="dense", op="dense", shape=(32,))  # Wrong shape
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_edge(input_node, dense_node)
        
        # Create debugger
        debugger = GraphDebugger()
        
        # Check graph for issues
        issues = debugger.check_graph(graph)
        self.assertGreater(len(issues), 0)
        
        # Test error handling
        with self.assertRaises(ValueError):
            graph.execute(np.random.randn(32, self.n_features))
            
    def test_graph_performance(self):
        """Test graph performance monitoring."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense_node = Node(name="dense", op="dense", shape=(64,))
        relu_node = Node(name="relu", op="relu", shape=(64,))
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_node(relu_node)
        graph.add_node(output_node)
        
        graph.add_edge(input_node, dense_node)
        graph.add_edge(dense_node, relu_node)
        graph.add_edge(relu_node, output_node)
        
        # Enable performance monitoring
        graph.enable_performance_monitoring()
        
        # Execute graph
        input_data = np.random.randn(32, self.n_features)
        graph.execute(input_data)
        
        # Get performance metrics
        metrics = graph.get_performance_metrics()
        self.assertIn("execution_time", metrics)
        self.assertIn("memory_usage", metrics)
        
    def test_graph_memory_management(self):
        """Test graph memory management."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense_node = Node(name="dense", op="dense", shape=(64,))
        relu_node = Node(name="relu", op="relu", shape=(64,))
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_node(relu_node)
        graph.add_node(output_node)
        
        graph.add_edge(input_node, dense_node)
        graph.add_edge(dense_node, relu_node)
        graph.add_edge(relu_node, output_node)
        
        # Enable memory monitoring
        graph.enable_memory_monitoring()
        
        # Execute graph
        input_data = np.random.randn(32, self.n_features)
        graph.execute(input_data)
        
        # Get memory metrics
        metrics = graph.get_memory_metrics()
        self.assertIn("peak_memory", metrics)
        self.assertIn("current_memory", metrics)
        
    def test_graph_parallel_execution(self):
        """Test graph parallel execution."""
        # Create graph
        graph = Graph(name="test_graph")
        
        # Add nodes
        input_node = Node(name="input", op="input", shape=(self.n_features,))
        dense_node = Node(name="dense", op="dense", shape=(64,))
        relu_node = Node(name="relu", op="relu", shape=(64,))
        output_node = Node(name="output", op="output", shape=(self.n_classes,))
        
        # Add nodes and edges
        graph.add_node(input_node)
        graph.add_node(dense_node)
        graph.add_node(relu_node)
        graph.add_node(output_node)
        
        graph.add_edge(input_node, dense_node)
        graph.add_edge(dense_node, relu_node)
        graph.add_edge(relu_node, output_node)
        
        # Enable parallel execution
        graph.enable_parallel_execution(num_threads=4)
        
        # Execute graph
        input_data = np.random.randn(32, self.n_features)
        output = graph.execute(input_data)
        
        # Verify execution
        self.assertEqual(output.shape, (32, self.n_classes))

if __name__ == '__main__':
    unittest.main() 