# tccg-6
import time
import tensor_comprehensions as tc
import torch

lang = """
def tccg6(float(G, E, B, C) L, float(D, F, G, A) R) -> (O) {
	O(a, b, c, d, e, f) +=! L(g, e, b, c) * R(d, f, g, a)
}
"""

print ("tccg-45: O(a, b, c, d, e, f) +=! L(g, e, b, c) * R(d, f, g, a)")
print ("(a,b,c,d,e,f,g) = (24,16,16,24,16,16,24)")
print ("# of Operations: ", 24*16*16*16*24*16*24)

matmul = tc.define(lang, name="tccg6")
mat1, mat2 = torch.randn(24, 16, 16, 16).cuda(), torch.randn(24, 16, 24, 24).cuda()

out = matmul(mat1, mat2)
torch.cuda.synchronize()

start = time.process_time()
for i in range(0, 10):
	out = matmul(mat1, mat2)
torch.cuda.synchronize()
end = time.process_time()

print ("time :", float(end - start) / 10)
