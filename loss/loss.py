from cmath import cos
import torch


class Loss():
    def __init__(self, mode):
        self.loss_MSE = torch.nn.MSELoss()
        self.loss_Cos = torch.nn.CosineSimilarity()
        self.mode = mode
        
    def __call__(self, rgbRaw, depthRaw, rgbPred, depthPred):
        if self.mode == 'MSE':
            loss = self.loss_MSE(rgbRaw, rgbPred) + self.loss_MSE(depthRaw, depthPred)
            return loss
        
        elif self.mode == 'Cos':
            loss = torch.mean(1 - self.loss_Cos(rgbRaw, rgbPred)) + torch.mean(1 - self.loss_Cos(depthRaw, depthPred))
            return loss
        
        elif self.mode == 'MSE_Cos':
            mseLoss = self.loss_MSE(rgbRaw, rgbPred) + self.loss_MSE(depthRaw, depthPred)
            cosLoss = torch.mean(1 - self.loss_Cos(rgbRaw, rgbPred)) + torch.mean(1 - self.loss_Cos(depthRaw, depthPred))
            loss = mseLoss + cosLoss   
            
            return loss, mseLoss, cosLoss   
              
