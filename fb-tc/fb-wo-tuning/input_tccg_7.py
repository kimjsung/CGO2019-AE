# tccg-7
import time
import tensor_comprehensions as tc
import torch

lang = """
def tccg7(float(G, F, A, B) L, float(D, E, G, C) R) -> (O) {
	O(a, b, c, d, e, f) +=! L(g, f, a, b) * R(d, e, g, c)
}
"""

print ("tccg-46: O(a, b, c, d, e, f) +=! L(g, f, a, b) * R(d, e, g, c)")
print ("(a,b,c,d,e,f,g) = (24,16,16,24,16,16,24)")
print ("# of Operations: ", 24*16*16*16*24*16*24)

matmul = tc.define(lang, name="tccg7")
mat1, mat2 = torch.randn(24, 16, 24, 16).cuda(), torch.randn(24, 16, 24, 16).cuda()
out = matmul(mat1, mat2)
torch.cuda.synchronize()
start = time.process_time()
for i in range(0, 10):
	out = matmul(mat1, mat2)
torch.cuda.synchronize()
end = time.process_time()
print ("time :", float(end - start) / 10)
