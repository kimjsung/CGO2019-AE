# tccg-8
import time
import tensor_comprehensions as tc
import torch

lang = """
def tccg8(float(G, F, A, C) L, float(D, E, G, B) R) -> (O) {
	O(a, b, c, d, e, f) +=! L(g, f, a, c) * R(d, e, g, b)
}
"""

print ("tccg-47: O(a, b, c, d, e, f) +=! L(g, f, a, c) * R(d, e, g, b)")
print ("(a,b,c,d,e,f,g) = (24,16,16,24,16,16,24)")
print ("# of Operations: ", 24*16*16*16*24*16*24)

matmul = tc.define(lang, name="tccg8")
mat1, mat2 = torch.randn(24, 16, 24, 16).cuda(), torch.randn(24, 16, 24, 16).cuda()
settings = {
"threads": 32, "generations": 20, "pop_size": 100, "number_elites": 10
}
torch.cuda.synchronize()
start = time.process_time()
matmul.autotune(mat1, mat2, cache=True,  **settings)
torch.cuda.synchronize()
end = time.process_time()
print ("autotune: ", end - start)
out = matmul(mat1, mat2)
start = time.process_time()
for i in range(0, 10):
	out = matmul(mat1, mat2)
torch.cuda.synchronize()
end = time.process_time()
print ("time :", float(end - start) / 10)
