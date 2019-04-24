class Node:
    '''
    Base class of all placeholder, variable and operation nodes.
    Every node should have a set of inputs, even if it is empty
    and a set of outputs, even if it is empty.
    '''
    
    def __init__(self):
        self.input_lst = [] # Initialized empty set of inputs
        self.output_lst = [] # Initialized empty set of outputs
        self.value = None # Initialized to None, after a forward pass
        # this value should be set to something
        
        # Add this node to currently activate graph
        _active_graph.append(self)
    
    def forward(self):
        '''
        Using input nodes' values it calculates the value of the node
        which will be used by next nodes as a input.
        '''
        pass
    
    def backward(self):
        '''
        Backward propagation. forward and backward function should
        be defined by each node separately, these empty definitions
        are just placeholders.
        '''
        pass
    

class Placeholder(Node):
    '''
    Represents a placeholder node which has only one job, holding a 
    single value. In forward pass, it just passes this value. Also
    note that, this value doesn't have to be provided when node is
    created, it can be given later on.
    '''
    def __init__(self):
        super().__init__()
    
    def set_value(self, value):
        self.value = value

class Variable(Node):
    '''
    Variable nodes and placeholder nodes are very similar. The only
    difference is that equations are differentiable with respect to
    a one or more variable.
    '''
    def __init__(self, initial_value=None):
        super().__init__()
        self.value = initial_value

    def set_value(self, value):
        self.value = value
        
class BinaryOperation(Node):
    '''
    Represent a binary operation on a computational graph. These nodes
    take two input (a, b) and calculates a single value in a forwar
    pass.
    '''
    def __init__(self, a, b):
        super().__init__()
        self.input_lst.append(a)
        self.input_lst.append(b)
        
        a.output_lst.append(self)
        b.output_lst.append(self)
        
    def partial_derivative(self, node):
        return None # All subclasses must implement this
    
    def backward(self):
        '''
        Chain rule calculation happens here. to calculate df/du we 
        multiple df/dv with dv/du.

        This function calculates each input's derivative according to
        chain rule
        '''

        # Chain rule
        for node in self.input_lst:
            node.derivative = self.partial_derivative(node) * self.derivative

class AddOperation(BinaryOperation):
    '''
    Plus Node, subclass of BinaryOperation. Sums two given input in the forward
    pass
    '''
    def forward(self):
        self.value = self.input_lst[0].value + self.input_lst[1].value
        
    def partial_derivative(self, node):
        return 1 # Partial derivative of d(x+y) w.r.t. x is 1.
    
class MultiplyOperation(BinaryOperation):
    '''
    (X) Node. Calculates a x b in the forward pass.
    '''
    def forward(self):
        self.value = self.input_lst[0].value * self.input_lst[1].value
    
    def partial_derivative(self, node):
        '''
        The function is the python equivalence of the following equations.

        d(a*b)/da = b
        d(a*b)/db = a
        '''
        a = self.input_lst[0]
        b = self.input_lst[1]
        
        other_node = a if node == b else b
        
        return other_node.value

# When constructing a computation graph, each node is added to active graph
# active graph is initialized to None here. When a graph is created and graph
# context is opened by using 'with' operator (with graph) active graph is set
# to that graph.
_active_graph = None

class Graph():
    '''
    Bundle class combining all nodes.
    '''
    
    def __init__(self):
        self.V = [] # This list stores all nodes added to the graph.
        self.order = None # This variable shows in which order nodes should be
        # be visited in forward pass. (will be filled in dfs method)
        
    def append(self, node):
        '''
        Add a node to graph's node container (self.V)
        '''
        
        if self.order:
            raise ValueError('Cannot add extra node after building the graph')
        
        self.V.append(node)
        node.visited = False
        
    def _calc_topological_order(self):
        '''
        Returns topological order for the current set of nodes
        It calculates this order by applying dfs on graph nodes.
        '''
        
        stack = []
        
        def dfs(node):
            # Depth first search
            
            for neighbor in node.output_lst:
                if not neighbor.visited:
                    dfs(neighbor)
            
            stack.append(node)
            node.visited = True
        
        for node in self.V:
            if not node.visited:
                dfs(node)
        
        # Stack is reverse of what we want so we reverse the list.
        self.order = stack[::-1]
        
    def forward(self):
        '''
        Forward pass of computation graph. What is does is simply
        calling forward method of nodes in a order given by
        topological order method.
        '''
        
        if not self.order:
            self._calc_topological_order()
        
        for node in self.order:
            node.forward()
        
        return self.order[-1].value
    
    def backward(self):
        '''
        Backward pass of computation graph. What is does is simply
        calling backward method of nodes in a reverse order given by
        topological order method. Since only Operations (or Equations)
        are differentiable, we call backward method on Operation
        nodes only.
        '''
        self.order[-1].derivative = 1
        for node in reversed(self.order):
            if isinstance(node, BinaryOperation):
                node.backward()
    
    def __enter__(self):
        '''
        Sets active_graph variable so that when a node is defined,
        it automatically added to active graph.
        '''
        global _active_graph
        _active_graph = self
        
    def __exit__(self, exc_type, exc_value, tb):
        global _active_graph
        _active_graph = None

if __name__ == '__main__':

    # J = d (a + bc)
    # J = d(a + u)
    # J = dv

    # Construct the graph

    graph = Graph()

    with graph: # Open graph context, defined nodes are placed there
        a = Variable() # Node a
        b = Variable() # Node b
        c = Variable() # Node c
        d = Variable() # Node d

        u = MultiplyOperation(b, c) # Node b * c
        v = AddOperation(a, u) # Node a + (b * c)

        J = MultiplyOperation(d, v) # node d * (a + b * c)
        
    # Set values

    # Setting values according to project description
    a.set_value(5)
    b.set_value(3)
    c.set_value(2)
    d.set_value(3)

    # Forward

    # Calling forward pass and printing the result
    result = graph.forward()

    print('Forward Propagation Result: %s' % result)

    # Backward

    # Calling backward pass and printing result for each variable
    graph.backward()

    print('dJ/da = %s' % a.derivative)
    print('dJ/db = %s' % b.derivative)
    print('dJ/dc = %s' % c.derivative)
    print('dJ/dd = %s' % d.derivative)