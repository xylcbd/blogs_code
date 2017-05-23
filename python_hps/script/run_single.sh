#!/bin/sh
#python
export PYTHONIOENCODING=utf-8
#start server
cd `pwd`/..
echo "run single pocess server"
python server.py
cd -
echo "server is started."
