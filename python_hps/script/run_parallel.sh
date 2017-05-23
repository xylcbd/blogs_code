#!/bin/sh
#python
export PYTHONIOENCODING=utf-8
#start server
echo "parallel server"
cd `pwd`/..
gunicorn -c gun.conf server:app
cd -
echo "server is started."
