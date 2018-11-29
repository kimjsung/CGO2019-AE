# tccg-4
import time
import tensor_comprehensions as tc
import torch

lang = """
def tccg4(float(G, E, A, B) L, float(D, F, G, C) R) -> (O) {
	O(a, b, c, d, e, f) +=! L(g, e, a, b) * R(d, f, g, c)
}
"""

print ("tccg-43: O(a, b, c, d, e, f) +=! L(g, e, a, b) * R(d, f, g, c)")
print ("(a,b,c,d,e,f,g) = (24,16,16,24,16,16,24)")
print ("# of Operations: ", 24*16*16*16*24*16*24)

matmul = tc.define(lang, name="tccg4")
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
