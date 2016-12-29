cmake_minimum_required(VERSION 2.8)

project(eyeTracker)

set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")

file(GLOB eyeTrackerSrc
	"*.h"
	"*.cpp"
)

link_directories(/opt/local/lib)
include_directories(/opt/local/include)

add_executable(eyeTracker ${eyeTrackerSrc})


target_link_libraries(eyeTracker freeImage)
