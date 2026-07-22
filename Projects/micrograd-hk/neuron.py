from value import Value
import random

class Neuron:
    def __init__(self , nin):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self ,X):
        out = sum([(w*x) for w ,x in zip(self.weights ,X) ],start=self.bias)
        out = out.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f'Neuron(weights = {self.weights}, bias= {self.bias})'
    
if __name__ == '__main__':
    neu = Neuron(3)
    X = [1.0,2.0,3.0]
    n = neu(X)
    print(f"""
neu = Neuron(3)
X = [1.0,2.0,3.0]
n = neu(X)
          
print(n) : {n}
parameters : {neu.parameters()}
""")