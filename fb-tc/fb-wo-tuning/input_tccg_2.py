# tccg-2
import time
import tensor_comprehensions as tc
import torch

lang = """
def tccg2(float(G, D, A, C) L, float(E, F, G, B) R) -> (O) {
	O(a, b, c, d, e, f) +=! L(g, d, a, c) * R(e, f, g, b)
}
"""

print ("tccg-41: O(a, b, c, d, e, f) +=! L(g, d, a, c) * R(e, f, g, b)")
print ("(a,b,c,d,e,f,g) = (24,16,16,16,24,16,24)")
print ("# of Operations: ", 24*16*16*16*24*16*24)

matmul = tc.define(lang, name="tccg2")
mat1, mat2 = torch.randn(24, 16, 24, 16).cuda(), torch.randn(24, 16, 24, 16).cuda()


out = matmul(mat1, mat2)
torch.cuda.synchronize()

start = time.process_time()
for i in range(0, 10):
	out = matmul(mat1, mat2)
torch.cuda.synchronize()
end = time.process_time()

print ("time :", float(end - start) / 10)
