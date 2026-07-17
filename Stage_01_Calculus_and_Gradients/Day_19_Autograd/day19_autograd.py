import numpy as np
import math 

class Value:
  def __init__(self, data , label = '' , _children =[], _op= '') -> None:
    self.data = data
    self.label = label
    self._prev = list(_children)
    self._op = _op
    self._backward = lambda : None
    self.grad  = 0.0


  def __add__(self ,other):
    other  =  other  if isinstance(other ,Value) else Value(other)
    out = Value(self.data + other.data , _children = [self, other], _op = '+')

    def _backward():
      self.grad  +=  1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __mul__(self ,other):
    other  =  other  if isinstance(other ,Value) else Value(other)
    out = Value(self.data * other.data , _children = [self, other], _op = '*')

    def _backward():
      self.grad  +=  other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __sub__(self,other):
    other =  other if isinstance(other , Value) else Value(other)
    out  = Value(self.data - other.data , _children=[self,other], _op = '-')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += -1.0 * out.grad
      

    out._backward = _backward

    return out 

  def __neg__(self):
    return -1 * self

  def __rmul__(self,other):
    return self * other 



  def backward(self):
    val = self
    val.grad = 1.0
    val_list = []
    def build_val_list(v):
      if v not in val_list:
        # val_list.append(v)
        for child in v._prev:
          build_val_list(child)

        val_list.append(v)

    build_val_list(self)

    for vals in reversed(val_list):
      print(vals)
      vals._backward()
    return val_list



  def sigmoid(self):
    sig  = 1/  ( 1 + math.exp(-self.data))
    out  = Value(sig , label = 'sigmoid' , _op= 'sig' , _children=[self])


    def _backward():
      print(sig)
      self.grad = sig * ( 1 - sig) * out.grad
      

    out._backward = _backward

    return out

  def __repr__(self) -> str:
    return f"Value(data = {self.data})"
  



a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c
d.backward()

print("a grad:", a.grad) # Should be -3.0
print("b grad:", b.grad) # Should be 2.0