# extractors/python_extractor.py

import os
import re
import ast
from collections import defaultdict
from core.feature_extractor import FeatureExtractor

class PythonExtractor(FeatureExtractor):
    """Feature extractor for Python code."""
    
    def __init__(self, project_path):
        """
        Initialize the Python feature extractor.
        
        Args:
            project_path (str): Path to the project root directory
        """
        super().__init__(project_path)
        self.file_extensions = ['.py']
    
    def extract_metrics_from_file(self, file_path):
        """
        Extract metrics from a Python file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Dictionary containing extracted metrics
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Initialize metrics with default values
        metrics = {metric: 0 for metric in self.get_all_metrics()}
        
        # Basic line-based metrics
        metrics['LOC'] = self._count_loc(content)
        
        # Try to parse the Python file using AST
        try:
            tree = ast.parse(content)
            self._extract_ast_metrics(tree, metrics)
        except SyntaxError as e:
            self.logger.warning(f"AST parsing failed for {file_path}: {str(e)}")
            # Fallback to regex-based metrics if parsing fails
            self._extract_basic_metrics(content, metrics)
        
        return metrics
    
    def _count_loc(self, content):
        """Count lines of code excluding comments and blank lines."""
        lines = content.split('\n')
        return len([line for line in lines if line.strip() and not line.strip().startswith('#')])
    
    def _extract_ast_metrics(self, tree, metrics):
        """Extract metrics using the Python AST."""
        
        class MetricsVisitor(ast.NodeVisitor):
            def __init__(self):
                self.class_count = 0
                self.function_count = 0
                self.method_count = 0
                self.field_count = 0
                self.return_count = 0
                self.loop_count = 0
                self.try_count = 0
                self.comparison_count = 0
                self.string_literal_count = 0
                self.number_literal_count = 0
                self.assignment_count = 0
                self.math_op_count = 0
                self.variable_count = 0
                self.function_calls = set()
                self.class_bases = defaultdict(list)
                self.current_class = None
                self.current_depth = 0
                self.max_depth = 0
                self.unique_words = set()
                
                # For CBO calculation
                self.imports = set()
                self.class_dependencies = defaultdict(set)
            
            def visit_ClassDef(self, node):
                self.class_count += 1
                old_class = self.current_class
                self.current_class = node.name
                
                # Record base classes for DIT calculation
                self.class_bases[node.name] = [base.id if isinstance(base, ast.Name) else base.__class__.__name__ for base in node.bases]
                
                # Visit class body
                self.generic_visit(node)
                self.current_class = old_class
            
            def visit_FunctionDef(self, node):
                if self.current_class:
                    self.method_count += 1
                else:
                    self.function_count += 1
                
                # Add function name to unique words
                self.unique_words.add(node.name)
                
                # Visit function body
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_Assign(self, node):
                self.assignment_count += len(node.targets)
                if self.current_class and isinstance(node.targets[0], ast.Name):
                    # Approximation for class fields
                    self.field_count += 1
                
                # Visit assigned values
                self.generic_visit(node)
            
            def visit_AnnAssign(self, node):
                self.assignment_count += 1
                if self.current_class and isinstance(node.target, ast.Name):
                    # Approximation for class fields with type annotation
                    self.field_count += 1
                
                # Visit assigned values
                self.generic_visit(node)
            
            def visit_Return(self, node):
                self.return_count += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.loop_count += 1
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_While(self, node):
                self.loop_count += 1
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_Try(self, node):
                self.try_count += 1
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_Compare(self, node):
                self.comparison_count += len(node.ops)
                self.generic_visit(node)
            
            def visit_Str(self, node):
                self.string_literal_count += 1
                self.generic_visit(node)
            
            def visit_Constant(self, node):
                # In Python 3.8+, literal values are represented as Constant nodes
                if isinstance(node.value, str):
                    self.string_literal_count += 1
                elif isinstance(node.value, (int, float, complex)):
                    self.number_literal_count += 1
                self.generic_visit(node)
            
            def visit_Num(self, node):
                # For Python < 3.8
                self.number_literal_count += 1
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                    self.math_op_count += 1
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.function_calls.add(node.func.id)
                    self.unique_words.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        # Record for CBO - calls to other classes
                        if self.current_class and node.func.value.id != 'self':
                            self.class_dependencies[self.current_class].add(node.func.value.id)
                        self.function_calls.add(f"{node.func.value.id}.{node.func.attr}")
                self.generic_visit(node)
            
            def visit_Name(self, node):
                self.unique_words.add(node.id)
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for name in node.names:
                    self.imports.add(name.name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    self.imports.add(node.module)
                self.generic_visit(node)
            
            def calculate_dit(self):
                """Calculate DIT (Depth Inheritance Tree) for each class."""
                dit_values = {}
                
                def get_dit(class_name, visited=None):
                    if visited is None:
                        visited = set()
                    
                    if class_name in visited:
                        return 0  # Avoid cycles
                    
                    visited.add(class_name)
                    
                    # extractors/python_extractor.py (continued)

                    if class_name not in self.class_bases or not self.class_bases[class_name]:
                        return 1  # Base class (no inheritance)
                    
                    max_depth = 0
                    for base in self.class_bases[class_name]:
                        # Skip non-tracked bases (built-ins, etc.)
                        if base in self.class_bases:
                            depth = get_dit(base, visited.copy())
                            max_depth = max(max_depth, depth)
                    
                    return max_depth + 1
                
                for class_name in self.class_bases:
                    dit_values[class_name] = get_dit(class_name)
                
                # Return the maximum DIT if there are any classes, otherwise 0
                return max(dit_values.values()) if dit_values else 0
        
        # Create and run the visitor
        visitor = MetricsVisitor()
        visitor.visit(tree)
        
        # Set metrics from visitor
        metrics['totalMethods'] = visitor.method_count + visitor.function_count
        metrics['totalFields'] = visitor.field_count
        metrics['returnQty'] = visitor.return_count
        metrics['loopQty'] = visitor.loop_count
        metrics['tryCatchQty'] = visitor.try_count
        metrics['comparisonsQty'] = visitor.comparison_count
        metrics['stringLiteralsQty'] = visitor.string_literal_count
        metrics['numbersQty'] = visitor.number_literal_count
        metrics['assignmentsQty'] = visitor.assignment_count
        metrics['mathOperationsQty'] = visitor.math_op_count
        metrics['variablesQty'] = visitor.assignment_count  # Approximation
        metrics['maxNestedBlocks'] = visitor.max_depth
        metrics['uniqueWordsQty'] = len(visitor.unique_words)
        
        # Calculate RFC (Response for Class)
        metrics['rfc'] = visitor.method_count + len(visitor.function_calls)
        
        # Calculate CBO (Coupling Between Objects)
        if visitor.class_count > 0:
            # Sum of unique dependencies across all classes
            all_deps = set()
            for deps in visitor.class_dependencies.values():
                all_deps.update(deps)
            metrics['CBO'] = len(all_deps) + len(visitor.imports)
        else:
            metrics['CBO'] = len(visitor.imports)
        
        # Calculate DIT (Depth Inheritance Tree)
        metrics['DIT'] = visitor.calculate_dit()
    
    def _extract_basic_metrics(self, content, metrics):
        """Extract basic metrics using regex patterns when AST parsing fails."""
        # Count lines of code
        metrics['LOC'] = self._count_loc(content)
        
        # Count class definitions
        class_matches = re.findall(r'class\s+\w+(\s*\([\w,\s]*\))?:', content)
        
        # Count function/method definitions
        function_matches = re.findall(r'def\s+\w+\s*\([^)]*\):', content)
        metrics['totalMethods'] = len(function_matches)
        
        # Count return statements
        metrics['returnQty'] = len(re.findall(r'\breturn\b', content))
        
        # Count loops
        metrics['loopQty'] = len(re.findall(r'\b(for|while)\b', content))
        
        # Count comparisons
        metrics['comparisonsQty'] = len(re.findall(r'(==|!=|<=|>=|<|>|is|is not|in|not in)', content))
        
        # Count try/except blocks
        metrics['tryCatchQty'] = len(re.findall(r'\btry\b', content))
        
        # Count string literals (both single and double quotes)
        metrics['stringLiteralsQty'] = len(re.findall(r'(\'[^\']*\'|"[^"]*")', content))
        
        # Count number literals
        metrics['numbersQty'] = len(re.findall(r'\b\d+(\.\d+)?\b', content))
        
        # Count assignments
        metrics['assignmentsQty'] = len(re.findall(r'=(?!=)', content))
        
        # Count math operations
        metrics['mathOperationsQty'] = len(re.findall(r'(\+|\-|\*|\/|\%|\*\*|\/\/|\<<|\>>|\&|\||\^)', content))
        
        # Approximate variable count
        variable_matches = re.findall(r'\b\w+\s*=', content)
        metrics['variablesQty'] = len(variable_matches)
        
        # Estimate fields (class variables)
        class_variable_matches = re.findall(r'self\.\w+\s*=', content)
        metrics['totalFields'] = len(class_variable_matches)
        
        # Estimate max nested blocks
        max_nested = 0
        current_nested = 0
        for line in content.split('\n'):
            indent_level = len(line) - len(line.lstrip())
            current_nested = indent_level // 4  # Assuming 4-space indentation
            max_nested = max(max_nested, current_nested)
        metrics['maxNestedBlocks'] = max_nested
        
        # Count unique words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        metrics['uniqueWordsQty'] = len(set(words))
        
        # Estimate RFC (Response for Class)
        method_calls = re.findall(r'\b\w+\.\w+\(', content)
        metrics['rfc'] = len(function_matches) + len(set(method_calls))
        
        # Estimate CBO (Coupling Between Objects)
        import_matches = re.findall(r'(import\s+[\w.]+|from\s+[\w.]+\s+import)', content)
        metrics['CBO'] = len(import_matches)
        
        # Estimate DIT (Depth Inheritance Tree)
        class_inheritance = re.findall(r'class\s+\w+\s*\([^)]+\):', content)
        metrics['DIT'] = 1 if class_inheritance else 0