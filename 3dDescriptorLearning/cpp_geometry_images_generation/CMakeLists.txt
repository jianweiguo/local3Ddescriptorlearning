cmake_minimum_required(VERSION 3.6)

project(GIGen)

# Note, replace these directories with the correct paths on your system
include_directories( ${CMAKE_CURRENT_LIST_DIR}\\include )
include_directories( C:\\dev\\OpenMesh-6.3\\include ) # OpenMesh head files directory. 
include_directories( C:\\Program\ Files\\MATLAB\\MATLAB\ Runtime\\v93\\extern\\include ) # MATLAB Runtime head files directory. 

link_directories( ${CMAKE_CURRENT_LIST_DIR}\\lib ) 
link_directories( C:\\dev\\OpenMesh-6.3\\lib ) # OpenMesh lib files directory. 
link_directories( C:\\Program\ Files\\MATLAB\\MATLAB\ Runtime\\v93\\extern\\lib\\win64\\microsoft ) # MATLAB Runtime lib files directory. 


aux_source_directory( ${CMAKE_CURRENT_LIST_DIR}/src src )

add_executable( ${PROJECT_NAME} ${src} )

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

  
target_link_libraries( ${PROJECT_NAME} debug  OpenMeshCored )
target_link_libraries( ${PROJECT_NAME} debug  OpenMeshToolsd )

target_link_libraries( ${PROJECT_NAME} optimized OpenMeshCore )
target_link_libraries( ${PROJECT_NAME} optimized OpenMeshTools )

target_link_libraries( ${PROJECT_NAME} libeng )
target_link_libraries( ${PROJECT_NAME} libmx )
target_link_libraries( ${PROJECT_NAME} libmat )
target_link_libraries( ${PROJECT_NAME} mclmcr )
target_link_libraries( ${PROJECT_NAME} mclmcrrt )
target_link_libraries( ${PROJECT_NAME} libcompcur )


add_definitions( -D_USE_MATH_DEFINES )

# set(CMAKE_CXX_FLAGS_DEBUG   "-O0" )            
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG " )   