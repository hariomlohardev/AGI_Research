from layer import Layer
from value import Value

class MLP:
    def __init__(self, nin , nouts: list):
        all = [nin] + nouts
        self.layers = [Layer(all[i] , all[i+1]) for i in range(len(all) -1)]
        
    def __call__(self, x : list):
        for l in self.layers:
            x = l(x)
        return x
    
    def zero_grad(self):
        for n in self.parameters():
            n.grad = 0.0
    
    def parameters(self) -> list[Value]:
        params  = []
        for lay in self.layers:
            params += lay.parameters()
        return params
    
    def __repr__(self):
        return f"MLP(layers={self.layers})"
    


if __name__ == '__main__':
    mlp = MLP(3 ,[3,3,1])
    X = [1.0,2.0,3.0]
    y_predicted = mlp(X)
    
