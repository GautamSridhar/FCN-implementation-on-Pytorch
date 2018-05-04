class Dice_loss(nn.Module):
    def __init__(self,type_weight=None,weights=None,ignore_index=None):
        super(Dice_loss,self).__init__()
        self.type_weight = type_weight
        self.weights=weights
        self.ignore_index=ignore_index

    def forward(output, target):
        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        type_weight : weights calculated according to the size of each segmented portion
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
        """
        Need to add ways to incorporate the weights that change according
        to the number of voxels attached to a particular label
        """
        eps = 0.0001

        encoded_target = output.detach() * 0

        if ignore_index is not None: 
            mask = target == ignore_index                        #creates a one hot encoding that masks a particular index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)   #creates a one hot encoding from the given target images   

        if weights is None:
            weights = 1

        ref_vol = encoded_target.sum(0)                          # creates an appropriate weight map that can be multiplied

        if type_weight == 'Square':
            weight_map = torch.reciprocal(ref_vol**2)
        if type_weight == 'Simple':
            weight_map = torch.reciprocal(ref_vol)

        new_weight_map = weight_map
        new_weight_map[weight_map == float("Inf")] = 0           # Converting all reciprocal infinities to max values
        m = torch.max(new_weight_map)
        new_weight_map[weight_map == float("Inf")] = m

        intersection = output * encoded_target
        numerator = 2 * (new_weight_map*intersection.sum(0)).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = (new_weight_map * denominator.sum(0)).sum(1).sum(1) + eps
        
        loss_per_channel = weights * (1 - (numerator / denominator))
        return loss_per_channel.sum() / output.size(1)