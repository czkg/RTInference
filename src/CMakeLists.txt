######### RTI Library ######### 
file(GLOB RTI_LIB_SOURCES ./*.cpp  ./*.c)

add_library(RTILibrary ${RTI_LIB_SOURCES})


set(TOOLS_LIBRARIES RTILibrary ${RTI_LIBRARIES})

set(TOOLS_LIBRARIES ${TOOLS_LIBRARIES} PARENT_SCOPE)
