import asyncio
import logging
import websockets
from websockets import WebSocketServerProtocol
from main import cap_height, cap_width, runModels
import json
import ssl
import pathlib

logging.basicConfig(level=logging.INFO)


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class Server:
    clients = set()

    async def register(self, ws: WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        logging.info(f'{ws.remote_address} connects.')
        async for message in AsyncIter([json.dumps({'Height': cap_height, 'Width': cap_width})]):
            await self.send_to_clients(message)

    async def unregister(self, ws: WebSocketServerProtocol) -> None:
        self.clients.remove(ws)
        logging.info(f'{ws.remote_address} disconnects.')

    async def send_to_clients(self, message: str) -> None:
        if self.clients:
            await asyncio.wait([asyncio.create_task(client.send(message)) for client in self.clients])

    async def distribute(self, ws: WebSocketServerProtocol) -> None:
        async for message in ws:
            res = runModels(json.loads(message)['msg'])
            await self.send_to_clients(json.dumps(res))

    async def ws_handler(self, ws: WebSocketServerProtocol, uri: str) -> None:
        await self.register(ws)
        try:
            await self.distribute(ws)
        finally:
            await self.unregister(ws)


# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# path_cert = pathlib.Path(__file__).with_name("cert.pem")
# path_key = pathlib.Path(__file__).with_name("key.pem")
# ssl_context.load_cert_chain(path_cert, keyfile=path_key)

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

# Generate with Lets Encrypt, chown to current user and 400 permissions
# ssl_cert = "F:\Atharva\depth_estimation_server\cert.pem"
# ssl_key = "F:\Atharva\depth_estimation_server\key.pem"

# ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)

server = Server()
# start_server = websockets.serve(
#     server.ws_handler, 'localhost', 8000, ping_interval=None, ssl=ssl_context)
start_server = websockets.serve(
    server.ws_handler, 'localhost', 8000, ping_interval=None)
loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(start_server)
    loop.run_forever()
except KeyboardInterrupt:
    pass


# # from aiohttp import web
# import socketio

# sio = socketio.Server(cors_allowed_origins='*')
# app = socketio.WSGIApp(sio)
# # app = web.Application()
# # sio.attach(app)

# # async def index(request):
# #     """Serve the client-side application."""
# #     with open('index.html') as f:
# #         return web.Response(text=f.read(), content_type='text/html')

# @sio.event
# def connect(sid, environ):
#     print("connect ", sid)

# @sio.event
# async def chat_message(sid, data):
#     print("message ", data)

# @sio.event
# def disconnect(sid):
#     print('disconnect ', sid)

# # app.router.add_static('/static', 'static')
# # app.router.add_get('/', index)

# # if __name__ == '__main__':
# #     web.run_app(app)