import torch
import torch.nn.functional as F
import torch.nn as nn

def cosine(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    similarity = torch.einsum("nc,ck->nk", [x, y.T])
    distances = 1 - similarity
    return distances

def euclidean(x, y):
    distances = torch.cdist(x, y, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    return distances

class tensor_center_loss:
    def __init__(
        self,  classes,device, fc_layer_dimension=32, beta=0.1,initial_value=None
    ):
        """
        This class implements center loss introduced in https://ydwen.github.io/papers/WenECCV16.pdf
        :param beta:  The factor by which centers should be updated
        :param classes: A list containing class labels for which we will be computing center loss \
                        Note-> This list should only contain positive numbers, negatives are reserved for unknowns
        :param fc_layer_dimension: The dimension of the layer in which center loss is being computed
        """
        self.beta = beta
        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2).to(device)
        self.cosine_dist_obj = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
        self.classs_num=len(classes)
        self.centers = torch.zeros((len(classes), fc_layer_dimension)).requires_grad_( requires_grad=False ).to(device)
        if initial_value is not None:
            for cls_no in classes:
                self.centers[cls_no] = torch.tensor(initial_value[cls_no]).to(device)

    def update_centers(self, FV, true_label):
        FV = FV.detach()
        deltas = FV - self.centers[true_label, :].requires_grad_(requires_grad=False)
        for cls_no in set(true_label.tolist()):
            #            print(self.centers[cls_no].shape,deltas[true_label==cls_no].shape,torch.mean(self.beta * deltas[true_label==cls_no],dim=0).shape)
            self.centers[cls_no] += self.beta * torch.mean(
                deltas[true_label == cls_no], dim=0
            )
    def __call__(self, FV, true_label):
        # Equation (2) from paper
        loss =  self.euclidean_dist_obj(FV, self.centers[true_label, :].to(FV.device))
        return loss.mean()


class Block(nn.Module):
    def __init__(self,  out_features, nfea,device):
        super(Block, self).__init__()
        self.U_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.U = nn.Linear(out_features, out_features).to(device)
        self.device = device

        self.R_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.R = nn.Linear(out_features, out_features).to(device)

        self.M_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.M = nn.Linear(nfea, nfea).to(device)
    def forward(self, input, view,E):

        D =self.M(self.M_norm(torch.mm(input.t(), view-E)))
        E = view - torch.mm(input, D)
        Z=self.R(self.R_norm(input))+ self.U(self.U_norm(torch.mm((view-E), D.t())))

        return Z,E
class FusionLayer(nn.Module):
    def __init__(self, num_views):
        super(FusionLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)

    def forward(self, emb_list,weight):
        common_emb = sum([w * emb_list[e] for w, e in zip(weight, emb_list.keys())])
        return common_emb

class Net(nn.Module):
    def __init__(self, nfeats, n_view,n_classes, para, device):
        super(Net, self).__init__()
        self.beta=0.1
        self.n_classes = n_classes

        self.device=device
        self.n_view=n_view
        self.nfeats=nfeats
        self.Blocks=nn.ModuleList([Block(n_classes,feat,device) for feat in nfeats])
        self.theta = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)
        self.theta2 = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)

        self.ZZ_init = nn.ModuleList([nn.Linear(feat,n_classes).to(device) for feat in nfeats])
        self.fusionlayer = FusionLayer(n_view)
        self.device=device
        self.centers = torch.zeros(self.n_classes,n_classes).requires_grad_(requires_grad=False).to(device)
        self.weight= torch.ones(self.n_view,1).to(device)

        self.centers_views = torch.zeros(self.n_view, self.n_classes, n_classes).requires_grad_(
            requires_grad=False).to(device)

        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2).to(device)

    def soft_threshold(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def soft_threshold2(self, u):
        return F.selu(u - self.theta2) - F.selu(-1.0 * u - self.theta2)
    def update_centers(self, FV, true_label):
        FV = FV[:true_label.shape[0]].detach()
        deltas = FV - self.centers[true_label, :].requires_grad_(requires_grad=False)
        for cls_no in set(true_label.tolist()):
            self.centers[cls_no] += self.beta * torch.mean(
                deltas[true_label == cls_no], dim=0
            )

    def update_view_centers(self, FV, true_label, view):
        FV = FV[:true_label.shape[0]].detach()
        deltas = FV - self.centers_views[view][true_label, :].requires_grad_(requires_grad=False)
        for cls_no in set(true_label.tolist()):
            self.centers_views[view][cls_no] += self.beta * torch.mean(
                deltas[true_label == cls_no], dim=0
            )

    def caculate_weight(self):

        d_views = torch.zeros(self.n_view, 1).to(self.device)
        for view in range(self.n_view):
            i = 0
            d_view = torch.zeros(int(self.n_classes * (self.n_classes - 1) / 2))
            for cls_no in range(self.n_classes - 1):
                for cls_no2 in range(cls_no + 1, self.n_classes):
                    d_view[i] = self.euclidean_dist_obj(self.centers_views[view][cls_no, :],
                                                        self.centers_views[view][cls_no2, :])
                    i += 1
            d_views[view] = min(d_view)
        d_views_ = torch.reciprocal(d_views)
        d_view_hat = d_views_ / torch.norm(d_views_, p=1, dim=0)

        self.weight = torch.exp(-d_view_hat) / torch.exp(-d_view_hat).sum()

    def forward(self, features, true_label,epoch):

        Ev= {}
        output_z = 0
        for j in range(self.n_view):
            output_z += self.ZZ_init[j](features[j] / 1.0)
            Ev[j] = torch.zeros(features[0].size()[0], self.nfeats[j]).to(self.device)
        output_z=output_z/self.n_view
        Z = {}
        for j in range(self.n_view):
            Z[j]=output_z


        for view in range(0,self.n_view):
            Z_view,E_view=self.Blocks[view](Z[view], features[view],Ev[view])
            Z[view]=self.soft_threshold(Z_view)
            Ev[view]=self.soft_threshold2(E_view)
            if epoch % 10 == 0:
                self.update_view_centers(Z[view], true_label, view)
        if epoch %10 == 0:
            self.caculate_weight()

        output_z = self.fusionlayer(Z, self.weight)
        self.update_centers(output_z, true_label)

        return output_z, self.centers
    def infer(self, features):

        Ev= {}
        output_z = 0
        for j in range(self.n_view):
            output_z += self.ZZ_init[j](features[j] / 1.0)
            Ev[j] = torch.zeros(features[0].size()[0], self.nfeats[j]).to(self.device)
        output_z = output_z / self.n_view
        Z = {}
        for j in range(self.n_view):
            Z[j] = output_z
        for view in range(0,self.n_view):
            Z_view,E_view=self.Blocks[view](Z[view], features[view],Ev[view])
            Z[view]=self.soft_threshold(Z_view)
            Ev[view] = self.soft_threshold2(E_view)
        output_z = self.fusionlayer(Z, self.weight)

        return output_z
