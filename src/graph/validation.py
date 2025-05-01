from typing import Dict, List, Optional, Any, Callable
import numpy as np
import time
import unittest
from . import Graph, ExecutionMode

class GraphValidator:
    """Validates computational graphs for correctness and performance."""
    
    def __init__(self):
        self._test_cases = []
        self._benchmarks = []
    
    def add_test_case(self, name: str, inputs: Dict[str, np.ndarray], expected_outputs: Dict[str, np.ndarray]) -> None:
        """Add a test case for graph validation."""
        self._test_cases.append({
            'name': name,
            'inputs': inputs,
            'expected_outputs': expected_outputs
        })
    
    def add_benchmark(self, name: str, inputs: Dict[str, np.ndarray], num_runs: int = 100) -> None:
        """Add a benchmark for performance testing."""
        self._benchmarks.append({
            'name': name,
            'inputs': inputs,
            'num_runs': num_runs
        })
    
    def validate(self, graph: Graph) -> Dict[str, Any]:
        """Validate the graph against all test cases."""
        results = {
            'passed': True,
            'test_results': [],
            'benchmark_results': []
        }
        
        # Run test cases
        for test_case in self._test_cases:
            test_result = self._run_test_case(graph, test_case)
            results['test_results'].append(test_result)
            if not test_result['passed']:
                results['passed'] = False
        
        # Run benchmarks
        for benchmark in self._benchmarks:
            benchmark_result = self._run_benchmark(graph, benchmark)
            results['benchmark_results'].append(benchmark_result)
        
        return results
    
    def _run_test_case(self, graph: Graph, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        result = {
            'name': test_case['name'],
            'passed': True,
            'errors': []
        }
        
        try:
            # Run in eager mode
            eager_outputs = graph.execute(test_case['inputs'])
            
            # Run in graph mode
            graph.compile(ExecutionMode.GRAPH)
            graph_outputs = graph.execute(test_case['inputs'])
            
            # Compare outputs
            for name, expected in test_case['expected_outputs'].items():
                if not np.allclose(eager_outputs[name], expected, rtol=1e-5, atol=1e-5):
                    result['passed'] = False
                    result['errors'].append(f"Output mismatch for {name}")
                
                if not np.allclose(graph_outputs[name], expected, rtol=1e-5, atol=1e-5):
                    result['passed'] = False
                    result['errors'].append(f"Graph output mismatch for {name}")
                
                if not np.allclose(eager_outputs[name], graph_outputs[name], rtol=1e-5, atol=1e-5):
                    result['passed'] = False
                    result['errors'].append(f"Eager/Graph output mismatch for {name}")
        
        except Exception as e:
            result['passed'] = False
            result['errors'].append(str(e))
        
        return result
    
    def _run_benchmark(self, graph: Graph, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark."""
        result = {
            'name': benchmark['name'],
            'eager_time': 0.0,
            'graph_time': 0.0,
            'speedup': 0.0
        }
        
        # Warm-up runs
        for _ in range(10):
            graph.execute(benchmark['inputs'])
        
        # Benchmark eager mode
        start_time = time.time()
        for _ in range(benchmark['num_runs']):
            graph.execute(benchmark['inputs'])
        result['eager_time'] = (time.time() - start_time) / benchmark['num_runs']
        
        # Benchmark graph mode
        graph.compile(ExecutionMode.GRAPH)
        start_time = time.time()
        for _ in range(benchmark['num_runs']):
            graph.execute(benchmark['inputs'])
        result['graph_time'] = (time.time() - start_time) / benchmark['num_runs']
        
        result['speedup'] = result['eager_time'] / result['graph_time']
        
        return result

class GraphTestSuite(unittest.TestCase):
    """Test suite for computational graphs."""
    
    def setUp(self):
        self.validator = GraphValidator()
    
    def test_graph_correctness(self):
        """Test graph correctness against test cases."""
        results = self.validator.validate(self.graph)
        self.assertTrue(results['passed'], f"Graph validation failed: {results}")
    
    def test_graph_performance(self):
        """Test graph performance against benchmarks."""
        results = self.validator.validate(self.graph)
        for benchmark in results['benchmark_results']:
            self.assertGreater(
                benchmark['speedup'],
                1.0,
                f"Graph mode is slower than eager mode for {benchmark['name']}"
            )

def create_benchmark_suite() -> unittest.TestSuite:
    """Create a benchmark test suite."""
    suite = unittest.TestSuite()
    # TODO: Add benchmark test cases
    return suite

def create_correctness_suite() -> unittest.TestSuite:
    """Create a correctness test suite."""
    suite = unittest.TestSuite()
    # TODO: Add correctness test cases
    return suite

# Global validator instance
validator = GraphValidator() 