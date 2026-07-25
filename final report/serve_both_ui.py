import os
import sys
import threading
import http.server
import socketserver
import json
import time

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path.startswith('/static/'):
            path = path[7:]
        return super().translate_path(path)

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == '/api/bins':
            data = [
                {'id': 'bin_0_0', 'label': 'BIN 0_0', 'layer': 0, 'col': 0, 'current': 2, 'total': 1, 'using': True, 'is_active': True, 'handedness': 'right', 'removed': 2, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 2, 'hand': True},
                {'id': 'bin_0_1', 'label': 'BIN 0_1', 'layer': 0, 'col': 1, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_0_2', 'label': 'BIN 0_2', 'layer': 0, 'col': 2, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_0_3', 'label': 'BIN 0_3', 'layer': 0, 'col': 3, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_0_4', 'label': 'BIN 0_4', 'layer': 0, 'col': 4, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_0_5', 'label': 'BIN 0_5', 'layer': 0, 'col': 5, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_1_0', 'label': 'BIN 1_0', 'layer': 1, 'col': 0, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_1_1', 'label': 'BIN 1_1', 'layer': 1, 'col': 1, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_1_2', 'label': 'BIN 1_2', 'layer': 1, 'col': 2, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
            ]
            self.send_json(data)
        elif self.path == '/api/layout':
            data = {
                'layers': [
                    {'layer': 0, 'row_slots': 6, 'bins': [
                        {'id': 'bin_0_0', 'slot_start': 0, 'span': 1, 'detected': True}, 
                        {'id': 'bin_0_1', 'slot_start': 1, 'span': 1, 'detected': True}, 
                        {'id': 'bin_0_2', 'slot_start': 2, 'span': 1, 'detected': True},
                        {'id': 'bin_0_3', 'slot_start': 3, 'span': 1, 'detected': True},
                        {'id': 'bin_0_4', 'slot_start': 4, 'span': 1, 'detected': True},
                        {'id': 'bin_0_5', 'slot_start': 5, 'span': 1, 'detected': True}
                    ]},
                    {'layer': 1, 'row_slots': 6, 'bins': [
                        {'id': 'bin_1_0', 'slot_start': 0, 'span': 2, 'detected': True}, 
                        {'id': 'bin_1_1', 'slot_start': 2, 'span': 2, 'detected': True}, 
                        {'id': 'bin_1_2', 'slot_start': 4, 'span': 2, 'detected': True}
                    ]}
                ]
            }
            self.send_json(data)
        elif self.path == '/api/stats':
            self.send_json({'fps': 30.0})
        elif self.path == '/api/alert':
            self.send_json({'active': True, 'message': 'WRONG BIN: Overpick detected in BIN 1 (Expected 1, got 2)'})
        elif self.path == '/api/kit':
            self.send_json({
                'active': 'bin_0_0',
                'done': [],
                'alert': {
                    'type': 'pick-from-wrong-bin',
                    'message': 'WRONG BIN: Overpick detected in BIN 1 (Expected 1, got 2)',
                    'bin': 'bin_0_0',
                    'bin_label': 'BIN 1'
                },
                'batch': {'done': 3, 'target': 10}
            })
        elif self.path == '/api/cycle':
            self.send_json({'set_number': 3, 'total_sets': 10, 'complete': False})
        elif self.path == '/state':
            self.send_json({'bins': {}, 'hands': [], 'errors': [], 'fps': 30, 'cycle': {}})
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass

def run_server(port, directory):
    class Handler(CustomHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('', port), Handler) as httpd:
        print(f"Serving {directory} on http://localhost:{port}")
        httpd.serve_forever()

if __name__ == '__main__':
    t1 = threading.Thread(target=run_server, args=(8080, r'C:\Users\yapor\OneDrive\Desktop\CDE3301\TEnterns\aegis-v2\integration\src\ui\static'), daemon=True)
    t2 = threading.Thread(target=run_server, args=(8081, r'C:\Users\yapor\OneDrive\Desktop\CDE3301\TEnterns\final report\sources\old_static'), daemon=True)
    t1.start()
    t2.start()
    while True:
        time.sleep(1)
