import json
import requests
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib import parse
import datetime


def post_result(request_id, src_num, dst_num):
    print("Start to post")
    print("This is RID: " + request_id)
    server_url = 'http://134.134.13.82:8744/imr-face-server/monitor/regmonitor'
    dic = {"ID": request_id, "Src_num": src_num, "Dest_num": dst_num}
    dic_json = json.dumps(dic)
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    print(str(datetime.datetime.now()))
    print(str(dic_json))
    response = requests.post(server_url, data=dic_json, headers=headers)
    print("Complete post")
    response.raise_for_status()
    print(response.content.decode('utf-8'))


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
        if type(ID) is list:
            ID = ID[0]
        if type(ID) is list:
            src_url = src_url[0]
        if type(ID) is list:
            dest_url = dest_url[0]
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
    last_time = datetime.datetime.now()

    def process(self):
        print(self.protocol_version)
        print(self.server_version)
        print(self.sys_version)
        print("process function hasn't been loaded yet.")

    def do_GET(self):
        if self.path == "/imr-monitor-server/parsevideo":
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
        if self.path == "/imr-monitor-server/parsevideo":
            print("postmsg recv, path right")
            data = self.rfile.read(int(self.headers["content-length"]))
            try:
                data = json.loads(data)
            except json.decoder.JSONDecodeError:
                data = parse.parse_qs(data.decode('utf-8'))
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            rspstr = "recv ok, data = "
            rspstr += json.dumps(data, ensure_ascii=False)
            self.wfile.write(rspstr.encode("utf-8"))
            if data.__contains__('ID') and data.__contains__('src_url') and data.__contains__('dest_url'):
                # region 这个要执行很久
                self.process_thread = ProcessThread(
                    data['ID'],
                    data['src_url'],
                    data['dest_url'],
                    self.process
                )
                if (datetime.datetime.now() - MyRequestHandler.last_time).seconds < 10:
                    print('too fast, wait 10 sec for next request.')
                while (datetime.datetime.now() - MyRequestHandler.last_time).seconds < 10:
                    pass
                self.process_thread.start()
                MyRequestHandler.last_time = datetime.datetime.now()
                # endregion
        elif self.path == "/imr-monitor-server/waitvideo":
            data = self.rfile.read(int(self.headers["content-length"]))
            data = json.loads(data)
            if data.__contains__('ID') and data.__contains__('Src_num') and data.__contains__('Dest_num'):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                rspstr = "recv ok, data = "
                rspstr += json.dumps(data, ensure_ascii=False)
                self.wfile.write(rspstr.encode("utf-8"))
            else:
                self.send_response(500)
                self.send_header("Content-type", "text/html")
                self.end_headers()
        else:
            print("postmsg recv, path error")
            self.send_response(500)
            self.send_header("Content-type", "text/html")
            self.end_headers()


if __name__ == '__main__':
    post_result(1, 2, 3)
