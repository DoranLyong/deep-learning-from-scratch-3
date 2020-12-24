import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator     # 1. Get a function (creator를 가져온다)
        if f is not None:
            x = f.input     # 2. Get the function's input (creator에서 input을 가져오고)
            x.grad = f.backward(self.grad)  # 3. Call the function's backward (함수의 backward 메서드 호출)
            x.backward()    # 4. (재귀) 하나 앞 변수의 backward 메서드를 호출한다 


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # Set parent(function)
        self.input = input
        self.output = output  # Set output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

    
if __name__ == "__main__":   

    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x) # a.data := 0.25
    b = B(a) # b.data := 1.2840254166877414
    y = C(b) # y.data := 1.648721270700128
    

    # backward
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad) # 3.297442541400256