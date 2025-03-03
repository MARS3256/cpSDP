# extractors/java_extractor.py

import os
import re
import javalang
from javalang.tree import ClassDeclaration, MethodDeclaration, FieldDeclaration
from collections import defaultdict
from core.feature_extractor import FeatureExtractor

class JavaExtractor(FeatureExtractor):
    """Feature extractor for Java code."""
    
    def __init__(self, project_path):
        """
        Initialize the Java feature extractor.
        
        Args:
            project_path (str): Path to the project root directory
        """
        super().__init__(project_path)
        self.file_extensions = ['.java']
    
    def extract_metrics_from_file(self, file_path):
        """
        Extract metrics from a Java file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Dictionary containing extracted metrics
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Initialize metrics with default values
        metrics = {metric: 0 for metric in self.get_all_metrics()}
        
        # Try to parse the Java file
        try:
            tree = javalang.parse.parse(content)
            self._extract_ast_metrics(tree, metrics, content)
        except Exception as e:
            self.logger.warning(f"AST parsing failed for {file_path}: {str(e)}")
            # Fallback to regex-based metrics if parsing fails
            self._extract_basic_metrics(content, metrics)
        
        return metrics
    
    def _extract_ast_metrics(self, tree, metrics, content):
        """Extract metrics using the Java AST."""
        # Count classes and interfaces
        class_count = 0
        
        # Track inheritance depth
        max_dit = 1  # All classes at least inherit from Object
        
        # Track method calls for RFC
        method_calls = set()
        
        # Track class dependencies for CBO
        dependencies = set()
        
        # Count fields and methods
        total_fields = 0
        total_methods = 0
        
        # Track other metrics
        return_count = 0
        loop_count = 0
        try_catch_count = 0
        variable_count = 0
        comparisons_count = 0
        parenthesized_count = 0
        string_literals = 0
        number_literals = 0
        assignments_count = 0
        math_operations = 0
        
        # Max nesting level
        max_nested_blocks = 0
        
        # Process the parse tree
        for _, class_decl in tree.filter(ClassDeclaration):
            class_count += 1
            
            # Calculate DIT
            current_dit = 1
            if class_decl.extends:
                current_dit += 1
                dependencies.add(class_decl.extends.name)
            
            # Record interfaces as dependencies for CBO
            if class_decl.implements:
                for interface in class_decl.implements:
                    dependencies.add(interface.name)
            
            # Record fields
            for _, field_decl in class_decl.filter(FieldDeclaration):
                total_fields += len(field_decl.declarators)
            
            # Process methods
            for _, method_decl in class_decl.filter(MethodDeclaration):
                total_methods += 1
                
                # Count method body elements
                if method_decl.body:
                    # Process method body for various metrics
                    # This would be more complex in a real implementation
                    pass
            
            max_dit = max(max_dit, current_dit)
        
        # Estimate LOC by removing comments and blank lines
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not (line.strip().startswith('//') or line.strip().startswith('/*') or line.strip().startswith('*'))]
        
        # Set the metrics
        metrics['CBO'] = len(dependencies)
        metrics['DIT'] = max_dit
        metrics['rfc'] = len(method_calls) + total_methods
        metrics['totalMethods'] = total_methods
        metrics['totalFields'] = total_fields
        metrics['LOC'] = len(non_empty_lines)
        
        # Further metrics would be extracted by a deeper AST analysis
        # Simplified for the example
    
    def _extract_basic_metrics(self, content, metrics):
        """Extract basic metrics using regex patterns when AST parsing fails."""
        # Count lines of code
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not (line.strip().startswith('//') or line.strip().startswith('/*') or line.strip().startswith('*'))]
        metrics['LOC'] = len(non_empty_lines)
        
        # Count method declarations
        method_matches = re.findall(r'(public|private|protected)?\s+\w+\s+\w+\s*\([^)]*\)\s*(\{|\s*throws)', content)
        metrics['totalMethods'] = len(method_matches)
        
        # Count field declarations
        field_matches = re.findall(r'(public|private|protected)?\s+(static)?\s+(final)?\s+\w+\s+\w+(\s*=\s*[^;]+)?;', content)
        metrics['totalFields'] = len(field_matches)
        
        # Count class declarations
        class_matches = re.findall(r'(public|private|protected)?\s+(abstract|final)?\s+class\s+\w+', content)
        
        # Estimate inheritance depth
        extends_matches = re.findall(r'extends\s+\w+', content)
        implements_matches = re.findall(r'implements\s+[\w,\s]+', content)
        metrics['DIT'] = 1 + (1 if extends_matches else 0)
        
        # Count dependencies for CBO
        import_matches = re.findall(r'import\s+[\w.]+;', content)
        metrics['CBO'] = len(import_matches)
        
        # Count method calls for RFC
        method_call_matches = re.findall(r'\b\w+\s*\([^)]*\)', content)
        metrics['rfc'] = len(method_call_matches)
        
        # Count return statements
        metrics['returnQty'] = len(re.findall(r'\breturn\b', content))
        
        # Count loops
        metrics['loopQty'] = len(re.findall(r'\b(for|while|do)\b', content))
        
        # Count comparisons
        metrics['comparisonsQty'] = len(re.findall(r'(==|!=|<=|>=|<|>)', content))
        
        # Count try/catch blocks
        metrics['tryCatchQty'] = len(re.findall(r'\btry\b', content))
        
        # Count parenthesized expressions (approximation)
        metrics['parenthesizedExpsQty'] = content.count('(')
        
        # Count string literals
        metrics['stringLiteralsQty'] = len(re.findall(r'"[^"]*"', content))
        
        # Count number literals
        metrics['numbersQty'] = len(re.findall(r'\b\d+\b', content))
        
        # Count assignments
        metrics['assignmentsQty'] = len(re.findall(r'=(?!=)', content))
        
        # Count math operations
        metrics['mathOperationsQty'] = len(re.findall(r'(\+|\-|\*|\/|\%|\<<|\>>|\&|\||\^)', content))
        
        # Count variable declarations
        variable_matches = re.findall(r'\b\w+\s+\w+\s*=', content)
        metrics['variablesQty'] = len(variable_matches)
        
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
        
        # Count unique words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        metrics['uniqueWordsQty'] = len(set(words))