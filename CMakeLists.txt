cmake_minimum_required(VERSION 3.10)
project(ML_Visualizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project source
set(PROJECT_SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

# Add ImGui implementation/source files from the included imgui folder
file(GLOB IMGUI_SOURCES
    ${PROJECT_SOURCE_DIR}/include/imgui/*.cpp
)
list(APPEND PROJECT_SOURCES ${IMGUI_SOURCES})

# Ensure the two backend files are present and added (explicit for clarity)
list(APPEND PROJECT_SOURCES
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_impl_glfw.cpp
    ${PROJECT_SOURCE_DIR}/include/imgui/imgui_impl_opengl3.cpp
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/imgui
    ${PROJECT_SOURCE_DIR}/include/GLEW
    ${PROJECT_SOURCE_DIR}/include/GLFW
)

# Find OpenGL
find_package(OpenGL REQUIRED)
if(TARGET OpenGL::GL)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENGL_gl_LIBRARY})
endif()

# Link GLEW and GLFW from the repo lib/ folder if present
if(EXISTS "${PROJECT_SOURCE_DIR}/lib/glew32.lib")
    # MSVC import library
    target_link_libraries(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/lib/glew32.lib")
elseif(EXISTS "${PROJECT_SOURCE_DIR}/lib/glew32.dll")
    # If only DLL present, copy at post-build. Linking to DLL directly isn't typical; prefer import lib
    message(STATUS "Found glew32.dll in lib/, you may need an import library for static linking")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/lib/libglfw3dll.a")
    # MinGW import library
    target_link_libraries(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/lib/libglfw3dll.a")
elseif(EXISTS "${PROJECT_SOURCE_DIR}/lib/glfw3.lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/lib/glfw3.lib")
elseif(EXISTS "${PROJECT_SOURCE_DIR}/lib/glfw3.dll")
    # Create an imported target for the DLL so it can be linked
    add_library(glfw3 SHARED IMPORTED)
    set_target_properties(glfw3 PROPERTIES IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/lib/glfw3.dll")
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw3)
else()
    message(STATUS "No bundled GLFW import library found in lib/. If you have system GLFW, consider using find_package(GLFW3 REQUIRED) or provide a library in lib/.")
endif()

# On Windows, also link to opengl32 (some generators/platforms require explicit link)
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE opengl32)
endif()

# MSVC conveniences
if(MSVC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Copy runtime DLLs (glfw/glew) to build output so executable can run next to them
if(WIN32)
  set(RUNTIME_DLLS)
  if(EXISTS "${PROJECT_SOURCE_DIR}/lib/glfw3.dll")
    list(APPEND RUNTIME_DLLS "${PROJECT_SOURCE_DIR}/lib/glfw3.dll")
  endif()
  if(EXISTS "${PROJECT_SOURCE_DIR}/lib/glew32.dll")
    list(APPEND RUNTIME_DLLS "${PROJECT_SOURCE_DIR}/lib/glew32.dll")
  endif()

  if(RUNTIME_DLLS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Copying runtime DLLs to output directory"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RUNTIME_DLLS} ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()
endif()

message(STATUS "Configured ${PROJECT_NAME}")
