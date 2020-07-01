
class ReplayBuffer:
    def __init__(self,  onlineBuffer, trainBuffer, offlineBuffer):
        self.onlineBuffer = onlineBuffer
        self.trainBuffer = trainBuffer
        self.offlineBuffer = offlineBuffer
   


    
    def getTrainBuffer(self):
        pass

    def pushOnlineBuffer(self):
        pass


    def pushOfflineBuffer(self):
        pass