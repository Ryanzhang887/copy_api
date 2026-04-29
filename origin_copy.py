import random

a = []
a = [0 if random.random() < 0.1 else 1 for _ in range(1024)]
    
print(a)