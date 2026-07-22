from nn import MLP
import math

class Loss:
    def __init__(self, mlp : MLP ,xs , ys):
        self.mlp = mlp 
        self.xs = xs
        self.ys = ys
        self.lr = 0.1
        self.loss_history = []

    def __call__(self, lr = 0.1 ,iterations = 20, optimizer:str = "adam", iter_stp = None):
        last_loss = 0
        mlp = self.mlp
        self.lr = lr
        self.t = 0
        
        

        for _ in range(iterations):
            
                
            y_pred = [mlp(x) for x in self.xs]
            loss  = sum((y_p - y_a)**2 for y_p , y_a in zip(y_pred , self.ys))
            self.loss_history.append(loss.data)

            for  n  in mlp.parameters():
                n.grad = 0.0
            
            loss.backward()

            self.optimizer(mlp , optimizer ,iter_stp)
            last_loss = loss.data

        return last_loss , y_pred
    
    def optimizer(self , mlp : MLP, optimizer :str ,iter_stp):
        if optimizer == "momentum":
            return self.momentum(mlp)
        elif optimizer == "SGD":
            return self.SGD(mlp)
        elif optimizer == "RSMProp":
            return self.RMSprop(mlp)
        elif optimizer == "adam_m":
            return self.adam_m(mlp , iter_stp = iter_stp)
        elif optimizer == "adam":
            return self.adam(mlp)
        


    
    def adam_m(self, mlp: MLP,iter_stp, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

    # 2. Check reset condition ONLY if iter_stp is explicitly provided
        if iter_stp is not None and self.t > iter_stp:
            self.t = 1  # Reset counter back to 1
            
        for n in mlp.parameters():
            n.momentum = n.momentum * beta1 + (2 - beta1) * n.grad
            n.velocity = (beta2 * n.velocity) + (2 - beta2) * (n.grad **2)

            n.momentum = n.momentum/(2 - beta1**self.t)
            n.velocity = n.velocity/(2 - beta2**self.t)

            n.data += -(self.lr * n.momentum)/math.sqrt(n.velocity + eps)

    def adam(self, mlp: MLP, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

        for n in mlp.parameters():
            n.momentum = n.momentum * beta1 + (1 - beta1) * n.grad
            n.velocity = (beta2 * n.velocity) + (1 - beta2) * (n.grad **2)

            momentum_hat = n.momentum/(1 - beta1**self.t)
            velocity_hat = n.velocity/(1 - beta2**self.t)

            n.data += -(self.lr * momentum_hat)/math.sqrt(velocity_hat + eps)



    def SGD(self , mlp:MLP) -> None:

        for n in mlp.parameters():
            n.data += -self.lr * n.grad

    
    def momentum(self , mlp : MLP , beta1=0.9) -> None:
        
        for n in mlp.parameters():
            n.momentum = n.momentum * beta1 + (1 - beta1) * n.grad
            n.data += -(self.lr * n.momentum)

    def RMSprop(self, mlp : MLP ,beta2=0.99, eps=1e-8) -> None:

        for n in mlp.parameters():
            n.velocity = (beta2 * n.velocity) + (1 - beta2) * (n.grad **2)
            n.data += -(self.lr * n.grad)/(math.sqrt(n.velocity + eps))

        
if __name__ == '__main__':
    # # 10 Data points in 2D space [x0, x1]
    # xs = [
    #     [0.5, 0.5],   # Class: 1.0
    #     [-0.5, -0.5], # Class: -1.0
    #     [0.8, -0.2],  # Class: 1.0
    #     [-0.8, 0.2],  # Class: -1.0
    #     [0.2, 0.8],   # Class: 1.0
    #     [-0.2, -0.8], # Class: -1.0
    #     [0.1, -0.5],  # Class: -1.0  <-- Non-linear twist!
    #     [-0.1, 0.5],  # Class: 1.0   <-- Non-linear twist!
    #     [0.7, 0.7],   # Class: 1.0
    #     [-0.7, -0.7]  # Class: -1.0
    # ]

    # ys = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0] # Target classes
    # xs = [[0.5, 0.5], [-0.5, -0.5], [0.8, -0.2], [-0.8, 0.2], [0.2, 0.8], 
    #   [-0.2, -0.8], [0.1, -0.5], [-0.1, 0.5], [0.7, 0.7], [-0.7, -0.7]]
    # ys = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]

    # 12 Data points in 2D space [x0, x1]
    # xs = [
    #     # Inner Circle (Class: 1.0) - Tiny radius (~0.2)
    #     [0.15, 0.15],   
    #     [-0.15, -0.15], 
    #     [0.20, -0.05],  
    #     [-0.20, 0.05],  
    #     [0.05, 0.20],   
    #     [-0.05, -0.20], 

    #     # Outer Circle (Class: -1.0) - Wider radius (~0.8)
    #     [0.60, 0.60],   
    #     [-0.60, -0.60], 
    #     [0.80, -0.10],  
    #     [-0.80, 0.10],  
    #     [0.10, 0.80],   
    #     [-0.10, -0.80]
    # ]

    # # Target classes matching the 12 coordinates above
    # ys = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    # 10 Data points in 2D space [x0, x1]
    # xs = [
    #     [0.1, 0.01],   # Class: 1.0 (On the sharp curve)
    #     [0.2, 0.04],   # Class: 1.0
    #     [0.3, 0.09],   # Class: 1.0
    #     [0.4, 0.16],   # Class: 1.0
    #     [0.5, 0.25],   # Class: 1.0
        
    #     # Tiny shifts off the curve (Class: -1.0)
    #     [0.1, 0.05],   # Class: -1.0 (Right above the curve!)
    #     [0.2, 0.08],   # Class: -1.0
    #     [0.3, 0.13],   # Class: -1.0
    #     [0.4, 0.20],   # Class: -1.0
    #     [0.5, 0.29]    # Class: -1.0
    # ]

    # ys = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]



    

    # m = MLP(2, [4, 4, 1])
    # loss_evaluator = Loss(m, xs, ys)
    # # Ensure you are calling self.adam() inside your Loss __call__ loop!
    # opt = "adam"
    # final_loss, y_pred = loss_evaluator(lr=0.25, iterations=200 , optimizer=opt) 
    # print("optimizer: ", opt)
    # print(f"Final Loss: {final_loss}")
    # for target, prediction in zip(ys, y_pred):
    #     print(f"Target: {target:+.1f} | Prediction: {prediction.data:+.3f}")


    # Normalized Real-World Housing Dataset (Scale: 0.0 to 1.0)
    xs = [
        [0.64, 0.45, 0.12],  # Premium -> Target:  1.0
        [0.59, 0.66, 0.01],  # Premium -> Target:  1.0
        [0.82, 0.39, 0.01],  # Premium -> Target:  1.0
        [0.38, 0.86, 0.16],  # Budget  -> Target: -1.0
        [0.34, 0.91, 0.16],  # Budget  -> Target: -1.0
        
        [1.00, 0.00, 0.03],  # Premium -> Target:  1.0
        [0.48, 0.81, 0.44],  # Budget  -> Target: -1.0
        [0.31, 0.85, 0.44],  # Budget  -> Target: -1.0
        [0.51, 0.73, 0.16],  # Premium -> Target:  1.0
        [0.00, 1.00, 1.00]   # Budget  -> Target: -1.0
    ]

    # Binary target classifications
    ys = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0]

    # Network Configuration
    m = MLP(3, [4, 4, 1])
    loss_evaluator = Loss(m, xs, ys)

    # Run Configuration
    opt = "adam_m"  # Swap between "adam" and "adam_m" to compare
    final_loss, y_pred = loss_evaluator(lr=0.02, iterations=250, optimizer=opt) 

    print("optimizer: ", opt)
    print(f"Final Loss: {final_loss}")
