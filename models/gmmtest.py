import dgl
import numpy as np
import torch as th
from dgl.nn import GMMConv
import torch
# g = dgl.graph(([0,1,2,3,2,5],[1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# feat = th.ones(6,10)
# conv = GMMConv(10, 2, 3, 2, 'mean')
# pseudo = th.ones(12,3)
# res = conv(g, feat, pseudo)
# print(res)

z = torch.randn(8,256,18)
eps = torch.bmm(z, z.transpose(2,1))
_, indices = torch.topk(eps, k=16, dim=1)
relation = torch.empty(8, 2, 16 * 256, dtype=torch.long)
relation[:, 0] = torch.Tensor(list(range(256)) * 16).unsqueeze(0).repeat(8,1)
relation[:, 1] = indices.view(8, -1)
represent = torch.randn(8,256,128)
u = relation[0][0].cuda()
v = relation[0][1].cuda()
feature = represent[0].cuda()
pseudo = th.ones(4096,3).cuda()
g = dgl.graph((v,u))
print(g.in_degrees())
GMM = GMMConv(128,128,3,25,'mean').cuda()
out = GMM(g, feature, pseudo)
print(out.shape)
