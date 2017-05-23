#!/bin/sh
curl --connect-timeout 5 -m 5 -s -X POST -H "Content-Type: application/json" -d @./post.data http://127.0.0.1:4096/api/foo
