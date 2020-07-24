from flask_classful import FlaskView
from getModell import create_app
import io
import zlib
from flask import Flask, request, Response, json
import numpy as np

SERVER_HOST= "localhost"
SERVER_PORT = 12345


app = create_app()
model = app.config['Model']


# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/storeqnetwork', methods=['POST'])
def storeNewQNetwork():
    data = request.json
   
    print("Store in online Buffer", data.keys())
    #state, action, camera, reward, next_state, next_camera, terminated
    q_network = data['qnetwork']

    print("Store q_network, length: ", len(q_network))
    for i in range(len(q_network)):
        q_network[i] = np.asarray(q_network[i])

    q_network = np.asarray(q_network)
    model.storeqNetworkWeights(q_network)
   
    
    return Response('',  status=200,
                    mimetype="application/octet_stream")


# route http posts to this method
@app.route("/getqnetwork", methods=['GET'])
def getQNetwork():

    q_network =  np.asarray(model.getQNetwork().get_weights())
    print(type(q_network), q_network.shape, type(q_network.tolist()))
    q_network_list = q_network.tolist()

    print(len(q_network_list))

    for i in range(len(q_network_list)):
        q_network_list[i] = q_network_list[i].tolist()
     
    print(q_network_list[0])

    data = {'qnetwork': q_network_list}
    

    return Response(response=json.dumps(data), status=200,
                    mimetype="application/json")

# route http posts to this method
@app.route("/gettargetnetworks", methods=['GET'])
def getTargetNEtworks():
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    r = request
    target1 =  np.asarray(model.getTarget1Network().get_weights())
    target1_list = target1.tolist()

    target2 =  np.asarray(model.getTarget2Network().get_weights())
    target2_list = target2.tolist()


    for i in range(len(target1_list)):
        target1_list[i] = target1_list[i].tolist()

    for i in range(len(target2_list)):
        target2_list[i] = target2_list[i].tolist()
  
    data = {'target1': target1_list,
     'target2': target2_list  }
    

    return Response(response=json.dumps(data), status=200,
                    mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True) 