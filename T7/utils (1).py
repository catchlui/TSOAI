## This function checks the accuracy of the prediction
from torchsummary import summary
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def display_model_summary(model,input_structure=(1,28,28)):
  summary(model, input_size=input_structure)