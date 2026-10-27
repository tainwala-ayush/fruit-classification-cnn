import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

TRAIN_DATA = r"D:\AI\CNN Classifier\FruitClassifier\Images\Train"
VAL_DATA = r"D:\AI\CNN Classifier\FruitClassifier\Images\Validation"
TEST_DATA = r"D:\AI\CNN Classifier\FruitClassifier\Images\Test"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_WORKER_STATE = True if NUM_WORKERS > 0 else False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 70


trainTransforms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale = (0.8,1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

valTransforms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
])

classLabels = ["Apple","Banana","Cherry","Coconut","Grapes","Orange","Pineapple","Strawberry","Watermelon"]
trainData = datasets.ImageFolder(TRAIN_DATA, trainTransforms)
valData = datasets.ImageFolder(VAL_DATA, valTransforms)
testData = datasets.ImageFolder(TEST_DATA, valTransforms)

trainDataLoader = DataLoader(dataset=trainData, batch_size = BATCH_SIZE, shuffle = True, num_workers= NUM_WORKERS, persistent_workers= NUM_WORKER_STATE, pin_memory= True)
valDataLoader = DataLoader(dataset=valData, batch_size= BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=NUM_WORKER_STATE, pin_memory= True)
testDataLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=NUM_WORKER_STATE, pin_memory=True)

def convBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout2d(0.2)

    )

num_classes = len(trainData.classes)

class fruitCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.features = nn.Sequential(
            convBlock(3,32),
            convBlock(32,64),
            convBlock(64,128),
            convBlock(128,256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        return self.classifier(self.features(x))


model = models.resnet18(weights= models.ResNet18_Weights.DEFAULT)
for params in model.parameters():
    params.requires_grad = False

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

lossFun = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.fc.parameters(), lr= 1e-3, weight_decay= 1e-5 )

def trainStep(trainDataLoader, model, optim, loss, num_classes = num_classes):
    trainLoss = 0
    trainAcc = MulticlassAccuracy(num_classes=num_classes, average = "macro").to(DEVICE)
    trainPrecision = MulticlassPrecision(num_classes=num_classes, average = "macro").to(DEVICE)
    trainRecall = MulticlassRecall(num_classes = num_classes, average = "macro").to(DEVICE)
    trainF1Score = MulticlassF1Score(num_classes = num_classes, average = "macro").to(DEVICE)

    model.train()

    for images, labels in trainDataLoader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        predicts = model(images)
        lossFn = loss(predicts, labels)
        optim.zero_grad()
        lossFn.backward()
        optim.step()

        predicted_class = predicts.argmax(dim=1)

        trainLoss += lossFn.item()
        trainAcc.update(predicted_class,labels)
        trainPrecision.update(predicted_class,labels)
        trainRecall.update(predicted_class,labels)
        trainF1Score.update(predicted_class,labels)

    trainLoss /= len(trainDataLoader)
    acc = trainAcc.compute().item()
    precision = trainPrecision.compute().item()
    recall = trainRecall.compute().item()
    f1Score = trainF1Score.compute().item()

    return trainLoss, acc, precision, recall, f1Score
        
def valStep(valDataLoader, model, lossFn, num_classes = num_classes):
    valLoss = 0
    valAcc = MulticlassAccuracy(num_classes=num_classes, average="macro").to(DEVICE)
    valPrecision = MulticlassPrecision(num_classes=num_classes, average = "macro").to(DEVICE)
    valRecall = MulticlassRecall(num_classes=num_classes, average = "macro").to(DEVICE)
    valF1Score = MulticlassF1Score(num_classes=num_classes, average = "macro").to(DEVICE)

    model.eval()   

    with torch.inference_mode():
        for images, labels in valDataLoader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            predict = model(images)

            loss = lossFn(predict, labels)

            prediction_classes = predict.argmax(dim=1)

            valLoss += loss.item()

            valAcc.update(prediction_classes, labels)
            valPrecision.update(prediction_classes, labels)
            valRecall.update(prediction_classes, labels)
            valF1Score.update(prediction_classes, labels)

    valLoss /= len(valDataLoader)
    acc = valAcc.compute().item()
    precision = valPrecision.compute().item()
    recall = valRecall.compute().item()
    f1Score = valF1Score.compute().item()

    return valLoss, acc, precision, recall, f1Score

def testStep(testDataLoader, model):
    testAcc = MulticlassAccuracy(num_classes=num_classes, average="macro").to(DEVICE)
    testPrec = MulticlassPrecision(num_classes=num_classes, average="macro").to(DEVICE)
    testRecall = MulticlassRecall(num_classes=num_classes, average="macro").to(DEVICE)
    testF1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(DEVICE)
    confusionMatrix = MulticlassConfusionMatrix(num_classes=num_classes).to(DEVICE)
    model.eval()

    with torch.inference_mode():
        for images, labels in testDataLoader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predict = model(images)
            prediction_classes = predict.argmax(dim=1)

            testAcc.update(prediction_classes, labels)
            testPrec.update(prediction_classes, labels)
            testRecall.update(prediction_classes, labels)
            testF1.update(prediction_classes, labels)
            confusionMatrix.update(prediction_classes,labels)

    acc = testAcc.compute().item()
    prec = testPrec.compute().item()
    recall = testRecall.compute().item()
    f1 = testF1.compute().item()
    confusionMatrix.compute()

    return acc, prec, recall, f1, confusionMatrix
    
class EarlyStopping():

    def __init__(self,patience, delta, path = r"D:\AI\CNN Classifier\FruitClassifier\Models\Checkpoints\best_path_2.pth"):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.path = path
        self.bestLoss = float("inf")

    def __call__(self, Loss, Acc, Precision, Recall, F1Score, model, optim, scheduler, epoch):
        if Loss < self.bestLoss - self.delta:
            self.bestLoss = Loss
            self.counter = 0
            torch.save({"model_state_dict": model.state_dict(), "Loss": Loss, "Accuracy": Acc, "Precision": Precision, "Recall": Recall, "F1Score": F1Score, "optim_state_dict": optim.state_dict(), "Epoch": epoch, "scheduler_state_dict": scheduler.state_dict()}, self.path)
            print(f"Model saved at {epoch} with a Loss of {Loss}")
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        return False

earlyStop = EarlyStopping(10,0.0001)
model = model.to(DEVICE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.1, patience=6, threshold=1e-8) 
def trainModel(model,trainDataLoader, valDataLoader, loss, optimizer, scheduler, earlyStop, epochs = EPOCH):
    results = {
        "trainLoss":[], "trainAcc":[], "trainPrec":[], "trainRecall":[], "trainF1":[],"valLoss":[], "valAcc":[], "valPrec":[], "valRecall":[], "valF1":[]
    }
    for i in range(100):
        if os.path.exists(r"D:\AI\CNN Classifier\FruitClassifier\Models\Checkpoints\best_path_2.pth"):
            model_checkpoint = torch.load(r"D:\AI\CNN Classifier\FruitClassifier\Models\Checkpoints\best_path_2.pth")
            model.load_state_dict(model_checkpoint["model_state_dict"])
            optimizer.load_state_dict(model_checkpoint["optim_state_dict"])
            scheduler.load_state_dict(model_checkpoint["scheduler_state_dict"])
            earlyStop.counter = 0
            print(f"Beginning from {model_checkpoint['Epoch']}")
            ep = model_checkpoint["Epoch"]
        else:
            print("Starting new training!")
            
        for ep in tqdm(range(epochs)):
            trainLoss, trainAcc, trainPrec, trainRecall, trainF1 = trainStep(trainDataLoader=trainDataLoader, model=model, optim=optimizer, loss=loss)
            valLoss, valAcc, valPrec, valRecall, valF1 = valStep(valDataLoader=valDataLoader, model=model, lossFn=loss)
            scheduler.step(valLoss)

            results["trainLoss"].append(trainLoss)
            results["trainAcc"].append(trainAcc)
            results["trainPrec"].append(trainPrec)
            results["trainRecall"].append(trainRecall)
            results["trainF1"].append(trainF1)

            results["valLoss"].append(valLoss)
            results["valAcc"].append(valAcc)
            results["valPrec"].append(valPrec)
            results["valRecall"].append(valRecall)
            results["valF1"].append(valF1)

            print(f"\nEpoch: {ep} | Train Loss: {trainLoss} | Train Accuracy: {trainAcc*100:.3f}% | Train Precision: {trainPrec*100:.3f}% | Train Recall: {trainRecall*100:.3f}% | Train F1 Score: {trainF1*100:.3f}% | Val Loss: {valLoss} | Val Accuracy: {valAcc*100:.3f}% | Val Precision: {valPrec*100:.3f}% | Val Recall: {valRecall*100:.3f}% | Val F1 Score: {valF1*100:.3f}% \n")

            if earlyStop(Loss = valLoss, Acc = valAcc, Precision= valPrec, Recall= valRecall, F1Score= valF1, model=model, epoch=ep, optim = optimizer, scheduler = scheduler):
                print("Early Stopped!")
                break
    return results

def confmatPlot(confmat, classLabels = classLabels):
    fig, ax = confmat.plot(labels = classLabels)
    plt.xticks(rotation=45)
    plt.show()
import matplotlib.pyplot as plt
import torch

def showMisclassified(model, testData, testDataLoader, device, maxImages=12):
    model.eval()
    mistakes = []

    with torch.inference_mode():
        for images, labels in testDataLoader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            wrong = predictions != labels

            for i in torch.where(wrong)[0]:
                mistakes.append((
                    images[i].cpu(),
                    labels[i].item(),
                    predictions[i].item()
                ))

                if len(mistakes) >= maxImages:
                    break

            if len(mistakes) >= maxImages:
                break

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    for ax, (image, actual, predicted) in zip(axes.flat, mistakes):
        # Undo normalization from [-1, 1] to [0, 1]
        image = (image * 0.5 + 0.5).clamp(0, 1)
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(
            f"Actual: {testData.classes[actual]}\n"
            f"Predicted: {testData.classes[predicted]}"
        )
        ax.axis("off")

    for ax in axes.flat[len(mistakes):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    result = trainModel(model = model, loss=lossFun, optimizer=optim, trainDataLoader=trainDataLoader, valDataLoader=valDataLoader, scheduler = scheduler, earlyStop = earlyStop)
    checkpoint = torch.load(r"D:\AI\CNN Classifier\FruitClassifier\Models\Checkpoints\best_path_2.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    testAcc, testPrec, testRecall, testF1,cm = testStep(model=model, testDataLoader=testDataLoader)
    #confmatPlot(cm)
    print(f"Test Accuracy: {testAcc * 100:.2f}%")
    print(f"Test Precision: {testPrec:.4f}")
    print(f"Test Recall: {testRecall:.4f}")
    print(f"Test F1 Score: {testF1:.4f}")
    showMisclassified(
    model=model,
    testData=testData,
    testDataLoader=testDataLoader,
    device=DEVICE
)