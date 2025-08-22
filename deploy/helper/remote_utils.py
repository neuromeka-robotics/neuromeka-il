import zmq
import pickle

        
class PickleServer:
    def __init__(self, host, port, request_handler):
        self.host = host
        self.port = port
        self.request_handler = request_handler

    def serve(self):
        # ZeroMQ server setup
        context = zmq.Context()
        socket = context.socket(zmq.REP)  # Reply socket for server
        socket.bind(f"tcp://{self.host}:{self.port}")
        print(f"Server listening on tcp://{self.host}:{self.port}")

        try:
            while True:
                # Wait for the client to send a message
                data = pickle.loads(socket.recv())
                response = self.request_handler.handle(data)
                socket.send(pickle.dumps(response))

        except KeyboardInterrupt:
            print("Server interrupted, shutting down...")

        finally:
            socket.close()
            context.term()
            print("Server shut down.")
            
            
class PickleClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

        # Initialize ZeroMQ context and socket in __init__
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)  # Request socket for client
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def send_data(self, data):
        self.socket.send(pickle.dumps(data))
        response = pickle.loads(self.socket.recv())
        return response

    def close(self):
        self.socket.close()
        self.context.term()
        
        
class BaseRequestHandler:
    def __init__(self, **kwargs):
        raise NotImplementedError
    
    def handle(self, received_data):
        raise NotImplementedError
    
    def success_response(self, message):
        message["result"] = "SUCCESS"
        return message
    
    def error_response(self, message):
        message["result"] = "ERROR"
        return message