#目标：在https://raw.githubusercontent.com/karpathy/makemore/master/names.txt这个names文件上，根据3个字母的context预测下一个字母
#目前参数配置达到的正确率为20%，较低的正确率的原因（1）模型不够好，可以后续加入Batch_Normalization层（2）参数没调过，lr偏高，最后loss在3震荡。解决方法是把lr调整成按训练进度减小
import torch
import torch.nn.functional as F

class BengioMLP:
    def __init__(self, vocab_size, emb_dim, block_size, hidden_dim):
        """
        vocab_size: 字符总数 (27)
        emb_dim: 嵌入维度 (比如 10)
        block_size: 上下文长度 (本任务中为 3)
        hidden_dim: 隐层神经元数量 (本任务：200)
        """
        self.block_size = block_size
        
        # 参数
        # 人工种子的意义是可以复现
        g = torch.Generator().manual_seed(2147483647)
        
        self.C = torch.randn((vocab_size, emb_dim), generator=g)
        
        # W1, b1: 线性层 1
        self.W1 = torch.randn((emb_dim * block_size, hidden_dim), generator=g) 
        self.b1 = torch.randn(hidden_dim, generator=g)
        
        # W2, b2: 线性层 2 (输出层)
        self.W2 = torch.randn((hidden_dim, vocab_size), generator=g)
        self.b2 = torch.randn(vocab_size, generator=g)
        
        # 引入梯度
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, x):
        emb = self.C[x] 
        
        emb_cat = emb.view(emb.shape[0], -1) #view是本节重点
        
        h = torch.tanh(emb_cat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def training_step(self, X, Y, lr=0.1):
        # 前向
        logits = self.forward(X)
        loss = F.cross_entropy(logits, Y)
        
        # 反向
        for p in self.parameters:
            p.grad = None
        loss.backward()
        
        # 梯度下降
        for p in self.parameters:
            p.data += -lr * p.grad
            
        return loss

#数据处理
import requests
url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
names = requests.get(url).text.splitlines()


#字母
chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

#字母编号化，取3个字母的context
def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size 
        for ch in w + '.':
            ix = stoi[ch]
            
            X.append(context) 
            Y.append(ix) 
            
            context = context[1:] + [ix] 
            
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

#参数设置
vocab_size = 27
emb_dim=10
block_size=3
hidden_dim=64
traing_batches = 1000
lr = 0.1

#主函数
if __name__ == "__main__":
    #训练集
    n1 = int(0.9*len(names))
    names_tr = names[:n1]
    names_ts = names[n1:]
    Xtr, Ytr = build_dataset(names_tr,block_size)
    model = BengioMLP(vocab_size, emb_dim, block_size, hidden_dim)
    
    for i in range(traing_batches):
        loss_tr = model.training_step(Xtr,Ytr)
        if i%10==9:
            print(f"Loss: {loss_tr:.4f}")
    
    #测试集
    Xts, Yts = build_dataset(names_ts,block_size)
    with torch.no_grad():
        logits_ts = model.forward(Xts)
        pred_y = torch.argmax(logits_ts, dim=1)
        acc = (pred_y == Yts).float().mean()
    loss_ts = model.training_step(Xts,Yts)
    print(f"Loss on test set:{loss_ts:.4f}, accuracy on test set:{acc}")
