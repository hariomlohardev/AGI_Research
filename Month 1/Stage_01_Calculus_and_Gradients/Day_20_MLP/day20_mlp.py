
import numpy as np
import matplotlib.pyplot as plt
import math
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
import random



from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._parant:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot



class Value:
    def __init__(self, data ,label= '', _childrens= [], _op= ''):
        self.data = data
        self.label = label
        self._parant = list(_childrens)
        self.grad = 0.0
        self._op = _op
        self._backward = lambda : None

    ###Operations
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data , _childrens=[self ,other] , _op = "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward =_backward

        return out

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out  =  Value(self.data * other.data , _childrens=[self,other], _op = '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        other = other if isinstance(other , (int , float)) else  'power function is only aplicable on an int or float value of power'
        out = Value(self.data ** other , _childrens=[self] ,_op = '**k')

        def _backward():
            self.grad += (other * (self.data ** (other -1))) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)

    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return -1.0 * self

    
    ###ReOperations
    def __radd__(self, other): # other + self 
        return self + other
    
    def __rsub__(self, other): # other - self
        return (-self) + other
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __rtruediv__(self, other): # other /self
        return self / other
    
    def __rpow__(self, other):
        return self ** other
    
    ## Functions like tanh()
    def tanh(self):
        num = math.exp(self.data) - math.exp(-self.data)
        deno = math.exp(self.data) + math.exp(-self.data)
        tanh =num/deno
        out = Value(tanh , _childrens=[self], _op = 'tanh')

        def _backward():
            tanh_grad = 1 - tanh **2
            self.grad += tanh_grad * out.grad
        out._backward = _backward

        return out 
    

    ## Backpropagation fn backward()
    def backward(self):
        self.grad = 1.0
        value = self.grad
        val_list  = []
        def gen_val_list(v):
            if v not in val_list:
                for child in v._parant:
                    gen_val_list(child)
                val_list.append(v)
                
        gen_val_list(self)
        for val in reversed(val_list):
            val._backward()



    
    def __repr__(self):
        return f'Value(data= {self.data})'


# a = Value(3.0)
# b = Value(2.0)
# c = a + b
# d  = a * c
# e = d **2
# f = 2 ** e
# f.backward()

# f,e,d,c



class Neuron:
    def __init__(self , nin):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(1,10))

    def __call__(self ,X):
        out = sum([(w*x) for w ,x in zip(self.weights ,X) ],start=self.bias)
        out = out.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f'Neuron(weights = {self.weights}, bias= {self.bias})'


class Layer:
    def __init__(self , nin , nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params += n.parameters()

        return params

        # return [n.parameters() for n in self.neurons]
        
    def __repr__(self):
        return f"Layer(neurons= {self.neurons})"


class MLP:
    def __init__(self, nin , nouts: list):
        all = [nin] + nouts
        self.layers = [Layer(all[i] , all[i+1]) for i in range(len(all) -1)]
        
    def __call__(self, x : list):
        for l in self.layers:
            out = l(x)
        return out
    
    def parameters(self):
        params  = []
        for lay in self.layers:
            params += lay.parameters()
        return params
    
    def __repr__(self):
        return f"MLP(layers={self.layers})"


lr = 0.05



class Loss:
    def __init__(self, mlp : MLP ,xs , ys):
        self.mlp = mlp 
        self.xs = xs
        self.ys = ys

    def __call__(self, lr = 0.1 ,iterations = 20):
        last_loss = 0
        mlp = self.mlp

        for _ in range(iterations):
            y_pred = [mlp(x) for x in self.xs]
            loss  = sum((y_p - y_a)**2 for y_p , y_a in zip(y_pred , self.ys))

            for  n  in mlp.parameters():
                n.grad = 0.0
            
            loss.backward()

            for n in mlp.parameters():
                n.data += -lr * n.grad

            last_loss = loss.data

        return last_loss , y_pred


xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
m = MLP(3, [3,3,1])



loss = Loss(m, xs ,ys)

loss(iterations=1000)
