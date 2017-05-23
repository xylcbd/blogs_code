#coding: utf-8
import flask
import json
import logging
import ctypes
import os

#python wrapper of libfoo
class FooWrapper:
    def __init__(self):
        cur_path = os.path.abspath(os.path.dirname(__file__))
        self.module = ctypes.CDLL(os.path.join(cur_path,'./impl/libfoo.so'))

    def foo(self,val):    
        self.module.foo.argtypes = (ctypes.c_int,)
        self.module.foo.restype = ctypes.c_int
        result = self.module.foo(val)
        return result

app = flask.Flask(__name__)
version = 'v0.1 2017/05/23 20:04'
fooWrapper = FooWrapper()

@app.route('/demo')
def demo():
    return app.send_static_file('demo.html')

@app.route('/api/version',methods=['GET','POST'])
def handle_api_version():
    return version

@app.route('/api/foo',methods=['GET','POST'])
def handle_api_foo():
    #get input
    val = flask.request.json['val']
    logging.info('[handle_api_foo] val: %d' % (val))
    #do calc
    result = fooWrapper.foo(val)
    logging.info('[handle_api_foo] result: %d' % (result))
    result = json.dumps({'result':result})
    return result

def setup_logging():
    logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s - %(levelname)s: %(message)s') 

setup_logging()
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4096,debug=True)
