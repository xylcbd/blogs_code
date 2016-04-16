@echo off
call build.bat
adb push libs/armeabi-v7a/neon_test /data/local/tmp/
adb shell chmod 777 /data/local/tmp/neon_test
adb shell /data/local/tmp/neon_test
pause