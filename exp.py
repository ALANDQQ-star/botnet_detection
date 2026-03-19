import torch
import torch.nn as nn
import numpy as np

# Given values from the challenge
n = 41
m = 15
q = 1000000007
Y = [776038603, 454677179, 277026269, 279042526, 78728856, 784454706, 29243312, 291698200, 137468500, 236943731, 733036662, 421311403, 340527174, 804823668, 379367062]

class ModuloNet(nn.Module):
    def __init__(self, n_in, m_out):
        super().__init__()
        
        self.conv = nn.Conv1d(1, 1, 3, stride=2, bias=False)
        conv_out_size = (n_in - 3) // 2 + 1
        
        self.fc = nn.Linear(conv_out_size, m_out, bias=False)

# Load the model
model = ModuloNet(n, m)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# Get weights
w_conv = model.conv.weight.squeeze().long().tolist()
w_fc = model.fc.weight.long().tolist()

print("Conv weights:", w_conv)
print("FC weights shape:", len(w_fc), "x", len(w_fc[0]))

# The forward pass is:
# conv_out[i] = sum(w_conv[j] * FLAG[i*2 + j] for j in range(3))
# for i in range(20) (since (41-3)//2 + 1 = 20)
# 
# Y[k] = (sum(w_fc[k][i] * conv_out[i] for i in range(20)) + noise) % q

# This creates a linear system. Let's build the matrix.
# Let FLAG be x[0], x[1], ..., x[40]
# conv_out[i] = w_conv[0]*x[2*i] + w_conv[1]*x[2*i+1] + w_conv[2]*x[2*i+2]
# 
# Y[k] = sum_{i=0}^{19} w_fc[k][i] * conv_out[i] + noise (mod q)
#      = sum_{i=0}^{19} w_fc[k][i] * (w_conv[0]*x[2*i] + w_conv[1]*x[2*i+1] + w_conv[2]*x[2*i+2])
#      = sum_{j=0}^{40} A[k][j] * x[j]  where A combines w_conv and w_fc

# Build matrix A (m x n) = (15 x 41)
A = np.zeros((m, n), dtype=np.int64)
for k in range(m):
    for i in range(20):  # 20 conv outputs
        for j in range(3):
            A[k, 2*i + j] += w_fc[k][i] * w_conv[j]

print("Matrix A shape:", A.shape)

# Now we have Y = A * x + noise (mod q)
# Since noise is small (-160 to 160), we can try to solve this

# The system is under-determined (15 equations, 41 unknowns)
# But we know FLAG bytes are printable ASCII (32-126)

# Let's try a lattice-based approach or brute force with constraints

# First, let's verify our understanding by checking if the fake flag works
fake_flag = b"SUCTF{fake_flag_xxx}"
print(f"\nFake flag length: {len(fake_flag)}")

# We need to solve for x where Y ≈ A * x (mod q)
# Since x values are small (32-126), we can try:
# 1. LLL lattice reduction
# 2. Constraint solving

# Let's try constraint solving with z3
from z3 import *

# Create solver
solver = Solver()

# Create variables for FLAG bytes
x = [Int(f'x_{i}') for i in range(n)]

# Add constraints: FLAG bytes are printable ASCII
for i in range(n):
    solver.add(x[i] >= 32)
    solver.add(x[i] <= 126)

# Add constraint for known prefix "SUCTF{" and suffix "}"
prefix = b"SUCTF{"
suffix = b"}"
for i, c in enumerate(prefix):
    solver.add(x[i] == c)
for i, c in enumerate(suffix):
    solver.add(x[n - 1 - i] == c)

# Add equations for Y
for k in range(m):
    # Y[k] = sum(A[k][j] * x[j]) + noise (mod q)
    # noise is in [-160, 160]
    expr = Sum([A[k, j] * x[j] for j in range(n)])
    # We need to handle modular arithmetic
    # Y[k] - expr - noise ≡ 0 (mod q)
    # Since noise is small, we can try: Y[k] - expr is in [-160, 160] mod q
    # Or equivalently: there exists noise in [-160, 160] such that (expr + noise) % q == Y[k]
    
    # Let's use: Y[k] - expr - noise ≡ 0 (mod q)
    # noise in [-160, 160]
    # (expr + noise) % q = Y[k] means expr + noise = Y[k] + k*q for some small k
    
    # Since FLAG bytes are small (32-126) and A values could be large,
    # expr could be any value. But the noise is small.
    
    # Let's try a different approach: for each Y[k], we need:
    # |(expr % q) - Y[k]| <= 160 (considering wrap around)
    
    # Actually, let's just check all possible noise values
    noise = Int(f'noise_{k}')
    solver.add(noise >= -160)
    solver.add(noise <= 160)
    
    # expr + noise ≡ Y[k] (mod q)
    # expr + noise - Y[k] = q * some_integer
    # We need to find if there exists an integer k such that expr + noise - Y[k] = q * k
    
    # Since expr can be large, let's compute bounds
    # For simplicity, let's try: expr + noise - Y[k] should be divisible by q
    
    solver.add((expr + noise - Y[k]) % q == 0)

print("Solving...")
if solver.check() == sat:
    model = solver.model()
    flag_bytes = bytes([model.eval(x[i]).as_long() for i in range(n)])
    print(f"FLAG: {flag_bytes.decode()}")
else:
    print("No solution found with basic approach")

# Let's also try a direct computation approach
# Since we know the structure, let's try to work backwards
print("\n--- Trying alternative approach ---")

# Actually, let me reconsider. The forward pass with real weights might give us more insight
# Let me verify by checking what Y we would get for the fake flag

def compute_forward(flag_bytes, w_conv, w_fc, add_noise=False):
    n = len(flag_bytes)
    x_data = list(flag_bytes)
    
    conv_out = []
    for i in range((n - 3) // 2 + 1):
        window = x_data[i*2 : i*2+3]
        val = sum(w * x for w, x in zip(w_conv, window))
        conv_out.append(val)
    
    Y = []
    for i in range(m):
        val = sum(w * x for w, x in zip(w_fc[i], conv_out))
        if add_noise:
            import random
            noise = random.randint(-160, 160)
        else:
            noise = 0
        Y.append((val + noise) % q)
    
    return Y, conv_out

# Let's check if we can find patterns
Y_test, conv_test = compute_forward(fake_flag, w_conv, w_fc, add_noise=False)
print(f"Y without noise for fake flag: {Y_test[:5]}...")
print(f"Given Y: {Y[:5]}...")

# The difference should give us information about noise
# But we need the real flag, not fake flag