# -*- coding: utf-8 -*-


import torchvision.models as models



model=models.vgg16(pretrained=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    
#define our new classifier
classifier = nn.Sequential(nn.Linear(25088, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                           
                                 nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                           
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                           
                                # nn.Linear(1568, 784),
                                # nn.ReLU(),
                                # nn.Dropout(0.2),
                           
                                 nn.Linear(512,102),
                                 #nn.ReLU(),
                                 #nn.Dropout(0.2),
                                 
                                 
                                 
                                 #nn.Linear(392,102),
                                 
                           
                                 nn.LogSoftmax(dim=1))

model.classifier=classifier

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
model.to(device);



epochs=10
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train()
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              #"Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
