# extractors/c_cpp_extractor.py

import os
import re
from collections import defaultdict
from core.feature_extractor import FeatureExtractor
from pycparser import c_parser, c_ast, parse_file
import logging

class CCppExtractor(FeatureExtractor):
    """Feature extractor for C/C++ code."""
    
    def __init__(self, project_path):
        """
        Initialize the C/C++ feature extractor.
        
        Args:
            project_path (str): Path to the project root directory
        """
        super().__init__(project_path)
        self.file_extensions = ['.c', '.cpp', '.h', '.hpp']
        self.parser = c_parser.CParser()
        
    def extract_metrics_from_file(self, file_path):
        """
        Extract metrics from a C/C++ file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Dictionary containing extracted metrics
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Basic metrics that can be extracted via regex
        metrics = self._extract_basic_metrics(content)
        
        # Try to parse the file for more complex metrics
        try:
            ast = parse_file(file_path, use_cpp=True)
            self._extract_ast_metrics(ast, metrics)
        except Exception as e:
            # Fallback to simplified AST-like metrics if parsing fails
            self.logger.warning(f"AST parsing failed for {file_path}: {str(e)}")
            self._extract_simplified_metrics(content, metrics)
        
        return metrics
    
    def _extract_basic_metrics(self, content):
        """Extract basic metrics using regex patterns."""
        metrics = {metric: 0 for metric in self.get_all_metrics()}
        
        # Count lines of code (excluding empty lines and comments)
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('//')]
        metrics['LOC'] = len(non_empty_lines)
        
        # Count string literals
        metrics['stringLiteralsQty'] = len(re.findall(r'"[^"]*"', content))
        
        # Count number literals
        metrics['numbersQty'] = len(re.findall(r'\b\d+\b', content))
        
        # Count return statements
        metrics['returnQty'] = len(re.findall(r'\breturn\b', content))
        
        # Count loops
        metrics['loopQty'] = len(re.findall(r'\b(for|while|do)\b', content))
        
        # Count comparisons
        metrics['comparisonsQty'] = len(re.findall(r'(==|!=|<=|>=|<|>)', content))
        
        # Count try/catch blocks
        metrics['tryCatchQty'] = len(re.findall(r'\btry\b', content))
        
        # Count math operations
        metrics['mathOperationsQty'] = len(re.findall(r'(\+|\-|\*|\/|\%|\<<|\>>|\&|\||\^)', content))
        
        # Count assignments
        metrics['assignmentsQty'] = len(re.findall(r'=(?!=)', content))
        
        # Count unique words (identifiers)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        metrics['uniqueWordsQty'] = len(set(words))
        
        return metrics
    
    def _extract_ast_metrics(self, ast, metrics):
        """Extract metrics using the AST (Abstract Syntax Tree)."""
        # This would be a complex visitor implementation to traverse the AST
        # Simplified for the example, but would be more comprehensive in practice
        class CMetricsVisitor(c_ast.NodeVisitor):
            def __init__(self):
                self.functions = []
                self.current_depth = 0
                self.max_depth = 0
                self.function_calls = set()
                self.fields = []
                self.variables = []
                self.parenthesized = 0
                
            def visit_FuncDef(self, node):
                self.functions.append(node.decl.name)
                self.generic_visit(node)
                
            def visit_StructOrUnion(self, node):
                if node.decl:
                    for decl in node.decl.type.decls:
                        self.fields.append(decl.name)
                self.generic_visit(node)
                
            def visit_Decl(self, node):
                if node.name:
                    self.variables.append(node.name)
                self.generic_visit(node)
                
            def visit_FuncCall(self, node):
                if hasattr(node.name, 'name'):
                    self.function_calls.add(node.name.name)
                self.generic_visit(node)
                
            def visit_Compound(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
            def visit(self, node):
                if isinstance(node, c_ast.ExprList):
                    self.parenthesized += 1
                super().visit(node)
        
        visitor = CMetricsVisitor()
        visitor.visit(ast)
        
        metrics['totalMethods'] = len(visitor.functions)
        metrics['totalFields'] = len(visitor.fields)
        metrics['rfc'] = len(visitor.function_calls)
        metrics['variablesQty'] = len(visitor.variables)
        metrics['maxNestedBlocks'] = visitor.max_depth
        metrics['parenthesizedExpsQty'] = visitor.parenthesized
        
        # These metrics are harder to get from C/C++ AST
        # Would require more complex analysis
        metrics['CBO'] = len(visitor.function_calls)  # Simplified approximation
        metrics['DIT'] = 1  # Default, would need class hierarchy analysis
        
    def _extract_simplified_metrics(self, content, metrics):
        """Extract simplified metrics when AST parsing fails."""
        # Count methods (functions)
        function_matches = re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*{', content)
        metrics['totalMethods'] = len(function_matches)
        
        # Estimate max nested blocks
        max_nested = 0
        current_nested = 0
        for char in content:
            if char == '{':
                current_nested += 1
                max_nested = max(max_nested, current_nested)
            elif char == '}':
                current_nested = max(0, current_nested - 1)
        metrics['maxNestedBlocks'] = max_nested
        
        # Count parenthesized expressions
        open_paren_count = content.count('(')
        metrics['parenthesizedExpsQty'] = open_paren_count
        
        # Estimate variables from declarations
        var_declarations = re.findall(r'\b(int|float|double|char|long|short|bool|void|unsigned|struct|enum)\s+\w+', content)
        metrics['variablesQty'] = len(var_declarations)
        metrics['totalFields'] = metrics['variablesQty']  # Simplified approximation
        
        # Approximation of RFC (response for class)
        function_calls = re.findall(r'\b\w+\(', content)
        metrics['rfc'] = len(set(function_calls))
        
        # Approximation of CBO (coupling between objects)
        metrics['CBO'] = metrics['rfc']  # Simplified approximation
        metrics['DIT'] = 1  # Default value