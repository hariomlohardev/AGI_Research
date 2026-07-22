import math 

class Value:
    def __init__(self, data ,label= '', _childrens= [], _op= '' ,momentum= 0.0 , velocity=0.0):
        self.data = data
        self.label = label
        self._parant = list(_childrens)
        self.grad = 0.0
        self._op = _op
        self._backward = lambda : None
        self.momentum = momentum
        self.velocity = velocity

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
    
    ## Functions like tanh()
    def tanh(self):
        if self.data >= 0:
            z = math.exp(-2.0 * self.data)
            tanh_val = (1.0 - z) / (1.0 + z)
        # If self.data is negative, calculate using positive exponents (safe since data is neg)
        else:
            z = math.exp(2.0 * self.data)
            tanh_val = (z - 1.0) / (z + 1.0)
        # num = math.exp(self.data) - math.exp(-self.data)
        # deno = math.exp(self.data) + math.exp(-self.data)
        # tanh =num/deno
        out = Value(tanh_val , _childrens=[self], _op = 'tanh')

        def _backward():
            tanh_grad = 1 - out.data **2
            self.grad += tanh_grad * out.grad
        out._backward = _backward

        return out 

    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0 , _childrens=[self] , _op = 'relu')

        def _backward():
            self.grad += (1.0 * out.grad) if self.data > 0 else 0

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
    

if __name__== '__main__':
    val1 = Value(3.0 , label= 'val1')
    val2 = Value(2.0, label='val2')
    print(f"""
Val1 = {val1.data} ,Val2 = {val2.data}
Addition : {val1 + val2}
Subtraction : {val1 - val2}
Multiplication : {val1 * val2}
division : {val1 / val2}
power : {val1 ** 2}
tanh : {val1.tanh().data}

""")