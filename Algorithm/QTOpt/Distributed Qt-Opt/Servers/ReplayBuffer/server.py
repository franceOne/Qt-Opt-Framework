from flask_classful import FlaskView
from getReplayBuffer import create_app
import io
import zlib
from flask import Flask, request, Response, json
import numpy as np

SERVER_HOST= "localhost"
SERVER_PORT = 12345


app = create_app()
replayBuffer = app.config['ReplayBuffer']

### HELPERS


# ## MAIN SERVER DESCRIPTOR/ROUTINE

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/storeonlinedata', methods=['POST'])
def storeOnlineData():
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    r = request
    data = request.json
    #
    #data = uncompress_nparr(r.data)
    #
    print("Store in online Buffer", data.keys())
    #state, action, camera, reward, next_state, next_camera, terminated
    state = np.array(data['state'])
    action = np.array(data['action'])
    camera = np.array(data['camera'])
    reward = data['reward']
    next_state = np.array(data['next_state'])
    next_camera = np.array(data['next_camera'])
    terminated = data['terminated']

    #print((state.shape), (action.shape), (camera.shape))
    #print((reward), (next_state.shape), (next_camera.shape), (terminated))
  
    replayBuffer.storeOnlineData( state, action, camera, reward, next_state, next_camera, terminated)
    print("After sored", replayBuffer.getOnlineBufferSize())
    
    return Response('',  status=200,
                    mimetype="application/octet_stream")

# route http posts to this method
@app.route('/storeofflinedata', methods=['POST'])
def storeOfflineData():
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    r = request
    data = request.json
    #
    #data = uncompress_nparr(r.data)
    #
    print("Store in offline Buffer", data.keys())
    #state, action, camera, reward, next_state, next_camera, terminated
    state = np.array(data['state'])
    action = np.array(data['action'])
    camera = np.array(data['camera'])
    reward = data['reward']
    next_state = np.array(data['next_state'])
    next_camera = np.array(data['next_camera'])
    terminated = data['terminated']
  
    replayBuffer.storeOfflineData(state, action, camera, reward, next_state, next_camera, terminated)
    print("After sored", replayBuffer.getOfflineBufferSize())
    
    return Response('',  status=200,
                    mimetype="application/octet_stream")

@app.route('/storetraindata/<batch_size>', methods=['POST'])
def storeTrainData(batch_size):
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    r = request
    data = request.json
    batch_size = int(batch_size)
    #
    #data = uncompress_nparr(r.data)
    #
    print("Store in Train Buffer", data.keys())
    #state, action, camera, reward, next_state, next_camera, terminated
    state = np.array(data['state'])
    action = np.array(data['action'])
    camera = np.array(data['camera'])
    reward = np.array(data['reward'])
    next_state = np.array(data['next_state'])
    next_camera = np.array(data['next_camera'])
    terminated = np.array(data['terminated'])
    q_target = np.array(data['q_target'])
  
    print(state.shape, action.shape, camera.shape, reward.shape, next_state.shape, next_camera.shape, terminated.shape, q_target.shape, )
    replayBuffer.storeTrainBuffer(state, action, camera, reward, next_state, next_camera, terminated, q_target, batch_size)
    print("After sored", replayBuffer.getTrainBufferSize())
    
    return Response('',  status=200,
                    mimetype="application/octet_stream")

# route http posts to this method
@app.route("/getOnlineBuffer/<batch_size>", methods=['GET'])
def getOnlineBuffer(batch_size):
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    batch_size = int(batch_size)
    r = request
    #
    #data = uncompress_nparr(r.data)
    #
    states, actions, cameras, next_states, next_cameras, rewards, terminates = replayBuffer.getOnlineBuffer(batch_size)
    data = {'state': np.asarray(states).tolist(), 'action': np.asarray(actions).tolist(), 
        'camera': np.asarray(cameras).tolist(), 'next_camera': np.asarray(next_cameras).tolist(), 'reward': np.asarray(rewards).tolist(), 'next_state': np.asarray(next_states).tolist(), 'terminated': np.asarray(terminates).tolist()  }
    
    print("Shapres", states.shape, actions.shape, cameras.shape, next_cameras.shape, next_states.shape)
    return Response(response=json.dumps(data), status=200,
                    mimetype="application/json")

# route http posts to this method
@app.route("/getOnlineBufferSize", methods=['GET'])
def getOnlineBufferSize():
    size = replayBuffer.getOnlineBufferSize().numpy()
    print("REplayBufferSize", size)    
    return Response(response=str(size), status=200)

# route http posts to this method
@app.route("/getTrainingBufferSize", methods=['GET'])
def getTrainBufferSize():
    size = replayBuffer.getTrainBufferSize().numpy()
    print("REplayBufferSize", size)    
    return Response(response=str(size), status=200)


@app.route("/getTrainBuffer/<batch_size>", methods=['GET'])
def getTrainBuffer(batch_size):
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    batch_size = int(batch_size)
    r = request
    #
    #data = uncompress_nparr(r.data)
    #
    states, actions, cameras, next_states, next_cameras, rewards, terminates, q_values = replayBuffer.getTrainBuffer(batch_size)
    data = {'state': np.asarray(states).tolist(), 'action': np.asarray(actions).tolist(), 
        'camera': np.asarray(cameras).tolist(), 'next_camera': np.asarray(next_cameras).tolist(), 'reward': np.asarray(rewards).tolist(),
        'next_state': np.asarray(next_states).tolist(), 'terminated': np.asarray(terminates).tolist(), 'q_value':  np.asarray(q_values).tolist() }
    
    print("Shapres", states.shape, actions.shape, cameras.shape, next_cameras.shape, next_states.shape)
    return Response(response=json.dumps(data), status=200,
                    mimetype="application/json")



if __name__ == '__main__':
    app.run(debug=True) 