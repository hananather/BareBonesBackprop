class Scalar:
    def __init__(self, data, grad=0.0):
        self.data = data  # The actual numerical value
        self.grad = grad  # The gradient of the value
        self._backward = lambda: None  # Function to backpropagate the gradient
        self._prev = set()  # Set of previous Scalars that this Scalar depends on
        self._op = ''  # The operation that produced this Scalar

    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"

    # Example of an operation: addition
    def __add__(self, other):
        out = Scalar(self.data + other.data)
        out._prev = {self, other}
        out._op = '+'

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    # Backpropagate the gradient
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# Example usage
if __name__ == "__main__":
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a + b
    c.backward()
    print(a, b, c)
