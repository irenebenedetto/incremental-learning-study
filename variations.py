import torch.nn as nn
import torch
from sklearn.svm import LinearSVC

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

def l2_loss(self, images, labels, old_net):
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
        dist_loss_l2 = l2_loss(fts_new, fts_old)*40
        
        loss = class_loss + dist_loss_l2

    return loss


def less_forget_constraint_loss(self, old_net, images, labels,num_old_classes, lambda_base=5, m=0.5, K=2):
    """
        Implementation of loss as .... paper

        Params:

        Return:
            loss, sum of distillation and classification loss
    """
    inputs = self.net(images)
    targets = labels
    # cross entropy loss for classification 
    loss_ce = nn.functional.cross_entropy(inputs, targets, reduction='none')

    # lambda parameter for less-forget constraint loss
    lmbd = lambda_base * np.sqrt((self.num_tot_classes - num_old_classes)/(num_old_classes))

    feature_old = old_net.feature_extractor(images)
    feature_new = self.net.feature_extractor(images)
    # less-forget constraint loss
    dist_loss = 1 - (feature_old * feature_new).sum(1)

    # inter-class separaton loss (already summed over exemplars of a batch)
    batch_size = images.size(0)
    loss_mr = 0
    for i, label in enumerate(labels):
        if label in range(num_old_classes):
            anchor = inputs/self.net.eta[i, label]
            inputs_new_classes = inputs[i, num_old_classes:]/self.net.eta
            topK_hard_negatives = torch.topk(inputs_new_classes, 2)[0]
            loss_mr += torch.nn.functional.margin_ranking_loss(anchor, topK_hard_negatives, target=torch.ones(K), margin=m, reduction='none').sum()

    # computing the total loss for the batch
    loss = (1/batch_size) * torch.sum(loss_ce + lmbd * dist_loss) + (1/num_old_classes) * loss_mr 
    return loss
        

# CLASSIFICATION VARIATION

def test_svc(self, test_dataset):
    self.net.eval()
    with torch.no_grad():
      test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
      running_corrects = 0
      t = self.num_tot_classes
      matrix = new_confusion_matrix(lenx=t, leny=t)
      tot_loss = 0
      for images, labels in test_dataloader:
        # print(f"Test labels: {np.unique(labels.numpy())}")
        images = images.to(self.DEVICE)
        fts_map = self.net.feature_extractor(images)
        # extract the prediction with svc
        preds = self.svc.predict(fts_map.cpu())
        update_confusion_matrix(matrix, torch.Tensor(preds).type_as(labels), labels)

        # Update Corrects
        running_corrects += torch.sum(torch.Tensor(preds) == labels.data).data.item()
      
      # Calculate Accuracy and mean loss
      accuracy = running_corrects / len(test_dataloader.dataset)
      print(f'\033[94mAccuracy on test set with svc :{accuracy}\x1b[0m')
      show_confusion_matrix(matrix)

def train_svc(self, X):
    """
        The fucntion perform the training, after the network training on the linear svc
        The svc classifier training function is called AFTER the training of the network
        with 
                            self.train_svc(dataloader)
        Params:
            X: the dataloader with new class images and exemplars
        Returns:
            scv trained
        
    """
    print('Training svc')
    with torch.no_grad():
      svc = LinearSVC()
      labels_list = []
      fts_list = []
      for images, labels in X:
        fts_map = self.net.feature_extractor(images.cuda())
        labels_list.append(labels)
        fts_list.append(fts_map.cpu())
        #svc.fit(fts_map.cpu(), labels)

      all_labels = torch.cat(labels_list)
      all_fts = torch.cat(fts_list)
      svc.fit(all_fts, all_labels)
      self.svc = svc
      return svc


