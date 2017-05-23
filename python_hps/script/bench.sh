#!/bin/sh
ab -T 'application/json' -p post.data -n 100 -c 10 http://127.0.0.1:4096/api/foo
