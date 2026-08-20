import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.exceptions
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    proxy=None,
                    ping_interval=60,
                    ping_timeout=600,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, TimeoutError, OSError, websockets.exceptions.WebSocketException) as exc:
                logging.info("Still waiting for server (%s: %s)...", type(exc).__name__, exc)
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except Exception:
            # Reconnect and retry once to survive transient socket timeouts.
            self._ws, self._server_metadata = self._wait_for_server()
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        control_msg = self._packer.pack({"__control__": "reset"})
        try:
            self._ws.send(control_msg)
            response = self._ws.recv()
        except Exception:
            # Reconnect and retry once.
            self._ws, self._server_metadata = self._wait_for_server()
            self._ws.send(control_msg)
            response = self._ws.recv()

        if isinstance(response, str):
            # Backward compatibility: old servers may not support reset control messages.
            logging.warning("Policy server reset is not supported; continuing without reset.")
            return

        result = msgpack_numpy.unpackb(response)
        if not isinstance(result, dict) or not result.get("ok", False):
            logging.warning("Unexpected reset response from policy server: %s", result)
