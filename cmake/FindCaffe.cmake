# Caffe package
unset(Caffe_FOUND)

###Set the variable Caffe_DIR as the root of your caffe directory
set(Caffe_DIR /home/cz/caffe)


find_path(Caffe_INCLUDE_DIRS NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/util/io.hpp caffe/layer.hpp
  HINTS
  ${Caffe_DIR}/include)


find_library(Caffe_LIBRARIES NAMES caffe
  HINTS
  ${Caffe_DIR}/build/lib)

message("lib_dirs:${Caffe_LIBRARIES}")

if(Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS)
    set(Caffe_FOUND 1)
endif()
