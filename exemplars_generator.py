from MLDL.nets.eg_resnet import EG_Resnet

def generate_images_with_network(self, label, n, X):
    """
    The function try to generate new images from a set of existing one, by 
    using a network with 3*32*32 output neurons

    Params:
    - X: set of images to replicate, they belong to the same class label
    - n: number of images to generate
    """
    print(f'generating +{n} new images of label {label}')
    net = EG_Resnet()
    net = net.cuda()
    l1_loss = nn.L1Loss()

    NUM_EPOCHS = 30

    BATCH_SIZE = 5
    LR = 0.01
    WEIGHT_DECAY = 5e-5

    MOMENTUM = 0.9
    GAMMA = 0.5
    MILESTONE = [int(NUM_EPOCHS/5), int(NUM_EPOCHS/3), int(NUM_EPOCHS/2)]

    # creating a dataset of random images
    ds_random_images = torch.randn(size=X.size())

    optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=GAMMA)
    
    # start straining to generate new images from X
    for epoch in range(NUM_EPOCHS):
        for random_image, x in zip(ds_random_images, X):
            net.train(True)
            random_image = random_image.cuda()
            output = net(random_image.unsqueeze(0)).squeeze()
            
            x = x.view(X.size()[1]*X.size()[2]*X.size()[3]).cuda()
            loss = l1_loss(output, x)
            # visualizing the evolution of an image
            #o = output.view(3, 32, 32)
            #plt.title(f'epoch {epoch}/{NUM_EPOCHS}, loss {loss.item()}')
            #plt.imshow(transforms.ToPILImage()(o.cpu()))
            #plt.show()

            loss.backward()
            optimizer.step()
            # --- end batch
        scheduler.step()

    # generating a new set of  n images initialized randomly
      
    with torch.no_grad():
        ds_new_random_images = torch.randn(size=(n, X.size()[1], X.size()[2], X.size()[3]))
        new_images = []
        net.eval()

        # feedforward the images to the network 
        for random_images in ds_new_random_images:
            random_images = random_images.cuda()
            outputs = net(random_images.unsqueeze(0)).squeeze().cpu()

            outputs = outputs.view(X.size()[1], X.size()[2], X.size()[3])
            new_images.append(outputs)

        new_images = torch.stack(new_images, dim=0)

        return new_images, net



def generate_new_image(self, label, n, X):
    print(f'generating +{n} images for label {label}')
    mean_of_X = X.mean(dim=0)
    std_of_X = X.std(dim=0)
    new_images = []
      
    for i in range(n):
        factor = np.random.random()*np.random.randint(-1, 2) 
        new_image = mean_of_X + factor*std_of_X
        new_images.append(new_image)

    return torch.stack(new_images)
