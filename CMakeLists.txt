cmake_minimum_required(VERSION 3.5)
project( CropImage )
find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
# Add a single include directory
include_directories(/usr/local/include)
add_executable( CropImage main.cpp )
target_link_libraries( CropImage ${OpenCV_LIBS} )
