import json
import requests
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler


def post_result(request_id, src_num, dst_num):
    server_url = 'http://localhost:8080'
    dic = {"ID": request_id, "Src_num": src_num, "Dest_num": dst_num}
    dic_json = json.dumps(dic)
    response = requests.post(server_url, data=dic_json)
    response.raise_for_status()
    print(response.content)


class ProcessThread(threading.Thread):
    ID = 0
    src_url = ""
    dst_url = ""

    def process(self):
        print(self.ID)
        print(self.src_url)
        print(self.dst_url)
        print("process function hasn't been loaded yet.")

    def __init__(self, ID, src_url, dest_url, process):
        threading.Thread.__init__(self)
        self.ID = ID
        self.src_url = src_url
        self.dst_url = dest_url
        self.process = process

    def run(self):
        self.process(self.ID, self.src_url, self.dst_url)


class MyRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "PSHS/0.1"
    sys_version = "Python/3.7.x"
    process_thread = None

    def process(self):
        print(self.protocol_version)
        print(self.server_version)
        print(self.sys_version)
        print("process function hasn't been loaded yet.")

    def do_GET(self):
        if self.path == "/" or self.path == "/index":
            print(self.path)
            req = {"success": "true"}
            self.send_response(200)
            self.send_header("Content-type", "json")
            self.end_headers()
            rspstr = json.dumps(req)
            self.wfile.write("You've gotten me.".encode("utf-8"))

        else:
            print("get path error")

    def do_POST(self):
        if self.path == "/signin":
            print("postmsg recv, path right")
        else:
            print("postmsg recv, path error")
            data = self.rfile.read(int(self.headers["content-length"]))
            data = json.loads(data)
            if data.__contains__('ID') and data.__contains__('src_url') and data.__contains__('dest_url'):
            # region 这个要执行很久
                self.process_thread = ProcessThread(data['ID'], data['src_url'], data['dest_url'], self.process)
                self.process_thread.start()
            # endregion
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            rspstr = "recv ok, data = "
            rspstr += json.dumps(data, ensure_ascii=False)
            self.wfile.write(rspstr.encode("utf-8"))
