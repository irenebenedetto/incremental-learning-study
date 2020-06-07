import torch.nn as nn
import torch

# LOSS VARIATION

 # Implementation of distillation loss as described by Hitlon et al., but most notably JEFF DEAN
def temperature_distillation_loss(self, images, labels, old_net, T=2, w_old_labels=0.5, w_distillation=2):
    """
    The function implement the distillation loss as described in paper ...
    Params:
        - images, labels: batch of training
        - old_net: old network frozen
        - T: temperature of distillation
        - w_old_labels: weight for classification loss
        - w_distillation: weight for distillation loss
    Return:
        the value of the loss 
    """
    outputs = self.net(images)[:, :self.num_tot_classes]
    num_old_classes = len(self.exemplar_sets)

    if num_old_classes == 0:
        loss = nn.functional.cross_entropy(outputs, labels)
        return loss

    # Compute input and target for distillation loss by using old newtork probabilities as target
    outputs_temperature = outputs[:, :num_old_classes]/T
    targets_temperature = old_net(images)[:,:num_old_classes]/T
    outputs_distillation = nn.functional.softmax(outputs_temperature, dim=1)
    targets_distillation = nn.functional.softmax(targets_temperature, dim=1)

    # For classification loss, all outputs of new network and corresponding one hot labels are used
    softmax_outputs_new_network = nn.functional.softmax(outputs, dim=1) # new network outputs
    labels_onehot = nn.functional.one_hot(labels, self.num_tot_classes).type_as(outputs)
    
    # Weight down classification loss on old outputs
    W_old_labels = torch.ones(num_old_classes) * w_old_labels
    W_distillation = torch.ones(num_old_classes) * w_distillation
    padded_W = torch.cat((W_distillation, W_old_labels, torch.ones(self.num_tot_classes-num_old_classes)))

    # Concatenate all values in a single negative log loss to compute total loss in one go
    loss_inputs = torch.cat((outputs_distillation, softmax_outputs_new_network), dim=1)
    loss_inputs = loss_inputs * padded_W.to(self.DEVICE)
    loss_targets = torch.cat((targets_distillation, labels_onehot), dim=1)
    loss = -loss_targets*torch.log(loss_inputs)
    return loss.sum(dim=1).mean()

def l2_loss(self, images, labels, old_net, dist_loss_weight=40):
    """
     The function compute the l2 loss as distillation loss and a cross entropy loss for the classification task
     Params:
         images: to classify
         labels
         num_old_classes: number of old classes 
         old_net: old network used to compute 
    Returns:
        TUA MAMMA               /the value of the total loss (distillation + classification)
    """
    outputs = self.net(images)[:, :self.num_tot_classes]
    num_old_classes = len(self.exemplar_sets)
      
    if num_old_classes == 0:
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(outputs, labels)

    else:
        l2_loss = nn.MSELoss()          # l2 loss
        CELoss = nn.CrossEntropyLoss()  # cross entropy loss

        fts_old = old_net.feature_extractor(images) 
        fts_new = self.net.feature_extractor(images)
        
        s = nn.Softmax(dim=1)
        
        class_loss = CELoss(outputs, labels)
        dist_loss_l2 = l2_loss(fts_new, fts_old)*dist_loss_weight
        
        loss = class_loss + dist_loss_l2

    return loss

def l1_loss(self, images, labels, old_net, dist_loss_weight=40):
    """
     The function compute the l2 loss as distillation loss and a cross entropy loss for the classification task
     Params:
         images: to classify
         labels
         num_old_classes: number of old classes 
         old_net: old network used to compute 
    Returns:
        the value of the total loss (distillation + classification)
    """
    outputs = self.net(images)[:, :self.num_tot_classes]
    num_old_classes = len(self.exemplar_sets)
      
    if num_old_classes == 0:
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(outputs, labels)

    else:
        l1_loss = nn.L1Loss()          # l1 loss
        CELoss = nn.CrossEntropyLoss()  # cross entropy loss

        fts_old = old_net.feature_extractor(images) 
        fts_new = self.net.feature_extractor(images)
        
        s = nn.Softmax(dim=1)
        
        class_loss = CELoss(outputs, labels)
        dist_loss_l1 = l1_loss(fts_new, fts_old)*dist_loss_weight
        
        loss = class_loss + dist_loss_l1

    return loss


def less_forget_loss(self, images, labels, old_net, lambda_base=2.5, m=0.5, K=2):
    inputs = self.net.forward_cosine(images)
    num_tot_classes = self.num_tot_classes
    num_old_classes = len(self.exemplar_sets)
    num_new_classes = num_tot_classes - num_old_classes

    targets = labels
    # cross entropy loss for classification 
    loss_ce = nn.functional.cross_entropy(inputs, targets, reduction='none')

    # Handle case for first batch
    if num_old_classes == 0:
        return loss_ce.mean()

    # lambda parameter for less-forget constraint loss
    lmbd = lambda_base * np.sqrt(num_new_classes/num_old_classes)

    feature_old = old_net.feature_extractor(images)
    feature_new = self.net.feature_extractor(images)
    # less-forget constraint loss
    dist_loss = 1 - (feature_old * feature_new).sum(1)

    # MATRIX ATTEMPT FOR LESS FORGETTING LOSS
    exemplar_idx = sum(labels.cpu().numpy() == label for label in range(num_old_classes)).astype(bool)
    exemplar_labels = labels[exemplar_idx].type(torch.long)
    anchors = inputs[exemplar_idx, exemplar_labels] / self.net.eta()
    out_new_classes = inputs[exemplar_idx, num_old_classes:] / self.net.eta()
    topK_hard_negatives, _ = torch.topk(out_new_classes, 2)
    # print(f'topK_hard_negatives shape: {topK_hard_negatives.shape}') #num of exemplars in batch
    loss_mr = torch.max(m - anchors.unsqueeze(1).to(self.DEVICE) + topK_hard_negatives.to(self.DEVICE), torch.zeros(1).to(self.DEVICE)).sum(dim=1).mean()

    # inter-class separaton loss (already summed over exemplars of a batch)
    batch_size = images.size(0)

    # computing the total loss for the batch
    loss = (1/batch_size) * torch.sum(loss_ce + lmbd * dist_loss) + loss_mr 
    return loss