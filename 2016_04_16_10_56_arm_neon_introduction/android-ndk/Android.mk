#traverse all the directory and subdirectory
define walk
  $(wildcard $(1)) $(foreach e, $(wildcard $(1)/*), $(call walk, $(e)))
endef

LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

#find all the file recursively
SRC_DIR := $(LOCAL_PATH)/../src
ALLFILES = $(call walk, $(SRC_DIR))
FILE_LIST := $(filter %.cpp, $(ALLFILES))

LOCAL_SRC_FILES := $(FILE_LIST)
LOCAL_ARM_NEON := true
LOCAL_CPPFLAGS := -fPIC -std=c++11 -mfloat-abi=softfp -mfpu=neon -march=armv7 -fno-tree-vectorize
LOCAL_LDFLAGS := -fPIE -pie
LOCAL_MODULE := neon_test

include $(BUILD_EXECUTABLE)
