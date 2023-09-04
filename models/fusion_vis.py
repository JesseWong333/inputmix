import torch
import torch.nn as nn
import torch.nn.functional as F
import vision_transformer as vits 
import utils

class FusionAttention(nn.Module):
    def __init__(self, attns) -> None:
        super().__init__()
        self.attn = attns
        self.scale = attns[0].scale
    
    def forward(self, x_l, externel_bool):
        if not externel_bool or len(x_l) == 1:
            return [ self.attn[i](x_l[i]) for i in range(len(x_l)) ]
        
        qkv_l = [ self.attn[i].cal_qkv(x_l[i]) for i in range(len(x_l)) ]
        
        summary_k = []
        summary_v = []
        for i in range(len(x_l)):
            k_l = []
            v_l = []
            for j in range(len(x_l)):
                if j != i:
                    k = qkv_l[j][1][:, :, 1:, :]
                    v = qkv_l[j][2][:, :, 1:, :]
                    k_l.append(k)
                    v_l.append(v)
            summary_k.append(torch.cat(k_l, 2))
            summary_v.append(torch.cat(v_l, 2))

        outputs = []
        for i in range(len(x_l)):
            out = self.attn[i].attend_with_summary_info(qkv_l[i][0], qkv_l[i][1], qkv_l[i][2], summary_k[i], summary_v[i])
            outputs.append(out)
    
        return outputs


class FusionBlock(nn.Module):
    def __init__(self, blocks) -> None:
        super().__init__()
        self.blocks = blocks
        self.attn = FusionAttention([ block.attn for block in self.blocks])

    def forward(self, x, externel):
        # y1, attn1, external_attn1, y2, attn2, external_attn2 = self.attn(self.blocks_rgb.norm1(x1), self.blocks_depth.norm1(x2), externel)
        outputs = self.attn( [ self.blocks[i].norm1(x[i]) for i in range(len(x))], externel )

        for i in range(len(x)):
            x[i] = x[i] + self.blocks[i].drop_path(outputs[i][0])
            x[i] = x[i] + self.blocks[i].drop_path(self.blocks[i].mlp(self.blocks[i].norm2(x[i])))
        return x

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class FusionTransformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.n_views = args.num_views
        
        base_modes = []
        heads = []
        for n_view in range(self.n_views):
            base_modes.append(vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0))
            embed_dim = base_modes[n_view].embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
            heads.append(LinearClassifier(embed_dim, num_labels=args.num_labels))
            utils.load_pretrained_weights(base_modes[n_view], "", args.checkpoint_key, args.arch, args.patch_size)

        fusion_blocks = []    
        for i in range(len(base_modes[0].blocks)):
            fusion_blocks.append(FusionBlock([model.blocks[i] for model in base_modes]))

        self.base_modes = nn.ModuleList(base_modes)
        self.heads = nn.ModuleList(heads)
        self.blocks = nn.ModuleList(fusion_blocks)


    def forward(self, x, fusion_layer=11, n=1):
        """
            x: list of views
        """
        for i in range(len(x)):
            x[i] = self.base_modes[i].prepare_tokens(x[i])

        outputs = []
        for i, blk in enumerate(self.blocks):
            externel = (i >= fusion_layer)
        
            x = blk(x, externel)   
          
            if len(self.blocks) - i <= n:
                # output.append((self.model_rgb.norm(x1), self.model_depth.norm(x2)))
                outputs.append([ self.base_modes[i].norm(x[i]) for i in range(len(x))])
        
        outputs = [torch.cat([layer_out[i][:, 0] for layer_out in outputs], dim=-1) for i in range(len(x))]

        outputs = [ self.heads[i](output) for i, output in enumerate(outputs)]
       
        return sum(outputs)
