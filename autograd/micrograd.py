#TASK1:从0构建automatic differentiation engine

#反向传播：1.节点（要本地导数和全局导数）2.建网络 3.应用
import math
import random

#brick：定义节点
class Value:
    def __init__(self,data,_children = (),_op = '',label = ''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda:None
    
    def __repr__(self):
        return f"Value(data = {self.data})"

    def __add__(self,other):
        if isinstance(other,Value) == False:
            other = Value(other)
        ans = Value(self.data+other.data,(self,other))
        def _backward():
            self.grad+=ans.grad
            other.grad+=ans.grad
        ans._backward = _backward
        ans._op = '+'
        return ans
    
    def __radd__(self,other):
        return self+other
    
    def __neg__(self): 
        return self * -1 

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self) 
    
    def __mul__(self,other):
        if isinstance(other,Value) == False:
            other = Value(other)
        ans = Value(self.data*other.data,(self,other))
        def _backward():
            self.grad+=other.data*ans.grad
            other.grad+=self.data*ans.grad
        ans._backward = _backward 
        ans._op = '*'
        return ans
    
    def __rmul__(self,other):
        return other*self


    def __pow__(self,x):
        ans = Value(self.data**x,(self,))
        def _backward():
            self.grad+=x*(self.data**(x-1))*ans.grad
        ans._backward = _backward
        ans._op = '**'
        return ans
    
    def __truediv__(self, other):
        # self / other  =>  self * (other**-1)
        return self * other**-1
    
    def __rtruediv__(self, other):
        # other / self  =>  other * (self**-1)
        return other * self**-1

    def tanh(self):
        ans = Value(math.tanh(self.data),(self,))
        def _backward():
            self.grad += (1-ans.data**2)*ans.grad
        ans._backward = _backward
        ans._op = 'tanh'
        return ans
    
    
    
    def backward(self):
        #拓扑结构的建立
        topo =  []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()

#搭建网络
class Neuron:
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w+[self.b]
    
class layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        # 优化：如果输出只有1维，直接返回标量，方便下一层处理
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        ans = []
        for neuron in self.neurons:
            ans +=neuron.parameters()
        return ans
    
class MLP:
    def __init__(self,nin,nouts):
        size = [nin] + nouts
        self.layers = [layer(size[i],size[i+1]) for i in range(len(nouts))]
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def _zerograd(self):
        for parameter in self.parameters():
            parameter.grad = 0


#实际应用：（gemini3出题）二维非线性二分类：MLP 拟合圆形边界。
import random
import math

def generate_donut_data(n_samples=50):
    xs = []
    ys = []
    
    for _ in range(n_samples):
        # 在 [-1, 1] x [-1, 1] 的区域内随机采样
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        
        radius_sq = x1**2 + x2**2
        
        # 半径小于 0.4 的是 +1
        # 半径大于 0.5 的是 -1 (留 0.1 的缝隙让边界清晰点)
        if radius_sq < 0.4:
            xs.append([x1, x2])
            ys.append(1.0)
        elif radius_sq > 0.5:
            xs.append([x1, x2])
            ys.append(-1.0)
            
    return xs, ys

# 生成 50 个样本
xs, ys = generate_donut_data(50)
print(f"Generated {len(xs)} samples.")

training_batches = 300
learning_rate = 0.05

model = MLP(2,[16,16,1])


#正式训练
for i in range(training_batches):
    ypred = [model(x) for x in xs]
    loss = sum((y-y_pred)**2/(len(xs)) for y,y_pred in zip(ys,ypred))
    loss.backward()
    for parameter in model.parameters():
        parameter.data -=parameter.grad*(learning_rate*(1-i/(1.2*training_batches)))
    model._zerograd()
    print(loss.data)

#TASK2:体验PyTorch的这个功能
import torch
import random
import math

#还是圆环分类问题
def generate_donut_data(n_samples=50):
    xs = []
    ys = []
    i = 0
    
    while i < n_samples:
        # 在 [-1, 1] x [-1, 1] 的区域内随机采样
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        
        # 计算半径平方
        radius_sq = x1**2 + x2**2
        
        # 定义分类逻辑：
        # 半径小于 0.4 的是 +1
        # 半径大于 0.5 的是 -1 (留 0.1 的缝隙让边界清晰点)
        if radius_sq < 0.4:
            xs.append([x1, x2])
            ys.append(1.0)
            i+=1
        elif radius_sq > 0.5:
            xs.append([x1, x2])
            ys.append(-1.0)
            i+=1
            
    return xs, ys

# 生成 50 个样本
xs, ys = generate_donut_data(50)
print(f"Generated {len(xs)} samples.")


X = torch.tensor(xs,dtype=torch.float32)
Y = torch.tensor(ys,dtype=torch.float32)

#print(X.shape,Y.shape)

Y = Y.view(-1,1)

#训练
W1 = (torch.randn(2, 16) * 0.1).requires_grad_(True)
b1 = (torch.randn(1, 16) * 0.1).requires_grad_(True)

W2 = (torch.randn(16, 16) * 0.1).requires_grad_(True)
b2 = (torch.randn(1, 16) * 0.1).requires_grad_(True)

W3 = (torch.randn(16, 1) * 0.1).requires_grad_(True)
b3 = (torch.randn(1, 1) * 0.1).requires_grad_(True)

W = [W1,W2,W3]
b = [b1,b2,b3]

#反向传播
num_batches = 2000
lr = 0.1

for i in range(num_batches):
    X1 = X@W1+b1
    X1 = X1.tanh()

    X2 = X1@W2+b2
    X2 = X2.tanh()

    Ypred = X2@W3+b3
    Ypred = Ypred.tanh()
    loss = (Ypred - Y).pow(2).mean()
    loss.backward()
    with torch.no_grad():
        for weight in W:
            weight.data -=lr*weight.grad
        for bias in b:
            bias.data -=lr*bias.grad
    
    for weight in W:
        weight.grad.zero_()
    
    for bias in b:
        bias.grad.zero_()
    #打印输出
    if i%100 == 99:
        print(f"{loss.data}")
