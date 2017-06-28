import numpy as np
import gc
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/(1024*1024)

print 'before load, memory: %d MB' % get_memory_usage()
#dataset = [np.zeros((6000,6000),dtype=np.float32) for _ in range(30000)]
dataset = [(np.zeros((6000,6000),dtype=np.float32),np.zeros((200,200),dtype=np.int32)) for _ in range(10000)]
print 'after load, memory: %d MB' % get_memory_usage()

print 'before release, memory: %d MB' % get_memory_usage()
del dataset
dataset = None
gc.collect()
print 'after release, memory: %d MB' % get_memory_usage()
