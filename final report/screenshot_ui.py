import os
import sys
import time
import threading
import http.server
import socketserver
import json
from playwright.sync_api import sync_playwright

PORT = 8080
DIRECTORY = sys.argv[1]
OUT_FILE = sys.argv[2]

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

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
                {'id': 'bin_0_0', 'label': 'BIN 1', 'layer': 0, 'col': 0, 'current': 2, 'total': 1, 'using': True, 'is_active': True, 'handedness': 'right', 'removed': 2, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 2, 'hand': True},
                {'id': 'bin_0_1', 'label': 'BIN 2', 'layer': 0, 'col': 1, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_0_2', 'label': 'BIN 3', 'layer': 0, 'col': 2, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_1_0', 'label': 'BIN 4', 'layer': 1, 'col': 0, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_1_1', 'label': 'BIN 5', 'layer': 1, 'col': 1, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_1_2', 'label': 'BIN 6', 'layer': 1, 'col': 2, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_2_0', 'label': 'BIN 7', 'layer': 2, 'col': 0, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_2_1', 'label': 'BIN 8', 'layer': 2, 'col': 1, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
                {'id': 'bin_2_2', 'label': 'BIN 9', 'layer': 2, 'col': 2, 'current': 0, 'total': 1, 'using': True, 'is_active': False, 'handedness': 'none', 'removed': 0, 'in_bom': True, 'detected': True, 'target': 1, 'placed': 0, 'hand': False},
            ]
            self.send_json(data)
        elif self.path == '/api/layout':
            data = {
                'layers': [
                    {'layer': 0, 'row_slots': 3, 'bins': [{'id': 'bin_0_0', 'slot_start': 0, 'span': 1, 'detected': True}, {'id': 'bin_0_1', 'slot_start': 1, 'span': 1, 'detected': True}, {'id': 'bin_0_2', 'slot_start': 2, 'span': 1, 'detected': True}]},
                    {'layer': 1, 'row_slots': 3, 'bins': [{'id': 'bin_1_0', 'slot_start': 0, 'span': 1, 'detected': True}, {'id': 'bin_1_1', 'slot_start': 1, 'span': 1, 'detected': True}, {'id': 'bin_1_2', 'slot_start': 2, 'span': 1, 'detected': True}]},
                    {'layer': 2, 'row_slots': 3, 'bins': [{'id': 'bin_2_0', 'slot_start': 0, 'span': 1, 'detected': True}, {'id': 'bin_2_1', 'slot_start': 1, 'span': 1, 'detected': True}, {'id': 'bin_2_2', 'slot_start': 2, 'span': 1, 'detected': True}]},
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

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('', PORT), CustomHandler) as httpd:
        httpd.serve_forever()

if __name__ == '__main__':
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    
    time.sleep(2)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({'width': 1200, 'height': 800})
        page.goto('http://127.0.0.1:8080')
        time.sleep(3) # Wait for fetch and render
        page.screenshot(path=OUT_FILE)
        browser.close()
    print('Saved', OUT_FILE)
    os._exit(0)
