from neuron import Neuron

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
    
    def __repr__(self):
        return f"Layer(neurons= {self.neurons})"