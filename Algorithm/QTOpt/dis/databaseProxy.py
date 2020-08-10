from flask_classful import FlaskView
from get_Database_Proxy import create_app
import io
import zlib
from flask import Flask, request, Response, json
import numpy as np

SERVER_HOST= "localhost"
SERVER_PORT = 12345


app = create_app()
database_adresses = app.config['database_adresses']

### HELPERS


def getRandomAress():
    index = np.random.randint(len(database_adresses), size=1)
    return database_adresses[index]

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
    url = getRandomAress()+"/storeonlinedata"

    try:
        requests.post(url, json=data)
        
    except:
        print("error store onlinedata")
        return Response('',  status=500,
                    mimetype="application/octet_stream")
   
    
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
    url = getRandomAress()+"/storeofflinedata"
    try:
            requests.post(url, json=data)
            
    except Exception as e:
            print("error store storeofflinedata", e)
            return Response('',  status=500,
                    mimetype="application/octet_stream")
    
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
    url = getRandomAress(url)+"/storetraindata/"+str(batch_size)
    try:
            requests.post(url, json=data)
            print("TrainData Stored")
    except Exception as e:
        print("error store traindata", e)
        return Response('',  status=500,
                    mimetype="application/octet_stream")
    
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
    url = getRandomAress()+"/getOnlineBuffer/"+str(batch_size)
    resp = requests.get(url).json()
    
    return Response(response=json.dumps(resp), status=200,
                    mimetype="application/json")

# route http posts to this method
@app.route("/getOnlineBufferSize", methods=['GET'])
def getOnlineBufferSize():
    url = getRandomAress()+"/getOnlineBufferSize"
    try:
            resp = requests.get(url).text
            print("getOnlineBufferSize", resp)
            return int(resp)
        except:
            print("Error fetching file")
    print("REplayBufferSize", size)    
    return Response(response=str(resp), status=200)

# route http posts to this method
@app.route("/getTrainingBufferSize", methods=['GET'])
def getTrainBufferSize():
    url = getRandomAress()+"/getTrainingBufferSize"
    try:
        resp = requests.get(url).text
        print("getTrainingBufferSize", resp)
        return int(resp)
    except:
        print("Error fetching file")
    
    return Response(response=str(resp), status=200)


@app.route("/getTrainBuffer/<batch_size>", methods=['GET'])
def getTrainBuffer(batch_size):
    """
    Expects a compressed, binary np array. Decompresses it, multiplies it by 10
    and returns it compressed.
    """
    batch_size = int(batch_size)
    r = request
    url = getRandomAress()+"/getTrainBuffer/"+str(batch_size)
    try:
        resp = requests.get(url).json()

        return Response(response=json.dumps(resp), status=200,
                mimetype="application/json")



if __name__ == '__main__':
    app.run(debug=True) 