Readme

The code is located in:
QTOpt/dis/*

The Code ist splitted in Backend and Frontend code
The Frontend requests the Data in the Backend over a HTTP Rest-Api

Servers:
In QTOpt/dis/Servers are two folders: Model and ReplayBuffer

The ReplayBuffer represents the Database in which the Buffers (Online-Buffer and Training-Buffer) stores the Data from the Qt-Opt Algorithm
The Model stores the DNN and updates it by a asynchronous Gradient descent

To run the Server Flesk is needed.
I deploy the Servers locally with following code:

export FLASK_APP=Servers/Model/serverModel
flask run  --port=5001


export FLASK_APP=Servers/ReplayBuffer/server
flask run  --port=5000

So per default the Replaybuffer will be deployed on port 5000 and the ServerModel on Port 5001


When the Servers are deployed, its possible to deploy the client.
For this you can for example run:

python3 Pendulum.py <datacollectorNum> <bellmanUpdaterNum> <trainingworkerNum>

Be sure that on the Servers are running the same Configuration and Environment like in the Client.

DataColector: Collects the Data from the Environment and stores it to the online Buffer
Bellmanupdater: Calculates the Q-Target Value and stores the transitions in the Train Buffer 
Trainingworker: Calculates the gradient for updating the DNN


Feel free :)

