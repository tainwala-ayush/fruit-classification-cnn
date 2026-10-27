import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix as cm
#Global Variables
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_WORKERS = 0
EPOCHS = 1000
PERSISTANT_WORKER_STATE = True if NUM_WORKERS > 0 else False
# Paths
trainPath = r"D:\AI\CNN Classifier\FruitClassifier\Images\Train"
valPath = r"D:\AI\CNN Classifier\FruitClassifier\Images\Validation"
testPath = r"D:\AI\CNN Classifier\FruitClassifier\Images\Test"

classLabels = ["Apple","Banana","Cherry","Coconut","Grapes","Mango","Orange","Pineapple","Strawberry","Watermelon"]
# Transfomers
trainTransformer = transforms.Compose([
                    transforms.Resize(size=(HEIGHT,WIDTH)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224,scale=(0.8,1)),
                    transforms.ColorJitter(brightness=0.2, contrast= 0.2),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean= [0.5,0.5,0.5],
                        std= [0.5,0.5,0.5]
                    )
])

validateTransformer = transforms.Compose([
                        transforms.Resize(size=(HEIGHT,WIDTH)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean = [0.5,0.5,0.5],
                            std = [0.5,0.5,0.5]
                        )
])

testTransformer = transforms.Compose([
                    transforms.Resize(size=(HEIGHT,WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean = [0.5,0.5,0.5],
                        std = [0.5,0.5,0.5]
                    )
])

# Datasets
trainData = datasets.ImageFolder(root = trainPath, transform=trainTransformer)
valData = datasets.ImageFolder(root = valPath, transform=validateTransformer)
testData = datasets.ImageFolder(root = testPath, transform=testTransformer)

# DataLoader
trainDataLoaded = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,persistent_workers=PERSISTANT_WORKER_STATE)
valDataLoaded = DataLoader(dataset=valData, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,persistent_workers=PERSISTANT_WORKER_STATE)
testDataLoaded = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,persistent_workers=PERSISTANT_WORKER_STATE)

def convBlock(input, output):
    return nn.Sequential(
        nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(output),
        nn.ReLU(),
        nn.Conv2d(in_channels=output, out_channels=output, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(output),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(0.2)
    )

class Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBlock1 = convBlock(3,32)
        self.convBlock2 = convBlock(32,64)
        self.convBlock3 = convBlock(64,128)
        self.convBlock4 = convBlock(128,256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256,10)
        )
    def forward(self, x):
        return self.classifier(self.convBlock4(self.convBlock3(self.convBlock2(self.convBlock1(x)))))

def trainStep(model:nn.Module, lossFun:nn.Module, optim:torch.optim.Optimizer, dataLoader:torch.utils.data.DataLoader,device = DEVICE):
    trainLoss, trainAcc = 0,0
    model.train()
    for i,(X,y) in enumerate(dataLoader):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = lossFun(y_pred,y)
        trainLoss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        y_class = torch.argmax(y_pred,dim=1)
        trainAcc += (y_class == y).float().mean().item()
    trainLoss /= len(dataLoader)
    trainAcc /= len(dataLoader)
    return trainLoss,trainAcc

def valStep(model:nn.Module, lossFun: nn.Module, dataLoader: torch.utils.data.DataLoader, device = DEVICE):
    valLoss,valAcc = 0,0
    model.eval()
    with torch.inference_mode():
        for i,(X,y) in enumerate(dataLoader):
            X,y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = lossFun(y_pred,y)
            valLoss += loss.item()
            y_class = torch.argmax(y_pred,dim=1)
            valAcc += (y_class==y).float().mean().item()
        valAcc /= len(dataLoader)
        valLoss /= len(dataLoader)
        return valLoss,valAcc

def testStep(model:nn.Module, dataLoader: torch.utils.data.DataLoader, device = DEVICE ):
    testAcc = 0
    confusionMatrix = cm(num_classes=10).to(device)
    model.eval()
    with torch.inference_mode():
        for i, (X,y) in enumerate(dataLoader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            acc = torch.argmax(y_pred, dim=1)
            confusionMatrix.update(acc,y)
            testAcc += (acc==y).float().mean().item()
    testAcc /= len(dataLoader)
    confusionMatrix.compute()
    return testAcc, confusionMatrix

class EarlyStopping():
    def __init__(self, patience, delta, path = "Checkpoints/best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.bestLoss = float("inf")
        self.counter = 0
        self.path = path
    def __call__(self, Loss, model:nn.Module, ep,optim,scheduler,Acc):
        if Loss < self.bestLoss - self.delta:
            self.bestLoss = Loss
            self.counter = 0
            torch.save({"Model":model.state_dict(),"epoch":ep,"Loss":Loss,"Accuracy":Acc,"Optimizer":optim.state_dict(), "Scheduler":scheduler.state_dict()},self.path)
            print(f"Model saved at Epoch: {ep} with Loss: {Loss} and Accuracy: {Acc}")
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        return False
earlyStop = EarlyStopping(patience=50, delta = 0.0001)
model1 = Model_1().to(DEVICE)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(),lr = 0.001, weight_decay=0.00001)   
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, threshold=1e-8)
def train(model:nn.Module, trainDataLoader: torch.utils.data.DataLoader, valDataLoader: torch.utils.data.DataLoader, loss: nn.Module, Optimizer: torch.optim.Optimizer, epochs = EPOCHS, device = DEVICE):
    results = {
        "trainLoss":[], "trainAcc":[], "valLoss":[], "valAcc":[]
    }
    for ep in tqdm(range(epochs)):
        trainLoss,trainAcc = trainStep(model=model, lossFun=loss, optim=Optimizer, dataLoader=trainDataLoader, device=device)
        valLoss, valAcc = valStep(model=model, lossFun=loss, dataLoader=valDataLoader, device=device)
        scheduler.step(valLoss)
        results["trainLoss"].append(trainLoss)
        results["trainAcc"].append(trainAcc)
        results["valLoss"].append(valLoss)
        results["valAcc"].append(valAcc)

        print(f"Epoch: {ep} | Train Loss: {trainLoss} | Train Accuracy: {trainAcc*100:.3f} | Validation Loss: {valLoss} | Validation Accuracy: {valAcc*100:.3f}\n")
        if earlyStop(Loss = valLoss, model=model1, ep=ep, optim = Optimizer, scheduler = scheduler, Acc=valAcc):
            print("Early Stopped!")
            break
    return results

def plotResults(result:dict):
    plt.figure(figsize=(10,7))
    epoch = range(len(result["trainLoss"]))
    plt.plot(epoch, result["trainLoss"], result["valLoss"])
    plt.plot(epoch, result["trainAcc"],result["valAcc"])
    plt.legend()
    plt.show()

def confmatPlot(confmat, classLabels = classLabels):
    fig, ax = confmat.plot(labels = classLabels)
    plt.xticks(rotation=45)
    plt.show()
if __name__ == "__main__": 
    #result = train(model = model1, loss=loss, Optimizer=optimizer, trainDataLoader=trainDataLoaded, valDataLoader=valDataLoaded)
    checkpoint = torch.load("Checkpoints/best_model.pth")
    model_loaded = Model_1().to(DEVICE)
    model_loaded.load_state_dict(checkpoint["Model"])
    testAcc, confmat = testStep(model=model_loaded, dataLoader=testDataLoaded)
    print(testAcc)
    confmatPlot(confmat)
    #plotResults(result)