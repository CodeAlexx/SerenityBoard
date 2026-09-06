cmake_minimum_required(VERSION 3.22)
project(serenityboard_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
enable_testing()

# Vendored SQLite (C, warnings not ours).
add_library(sb_sqlite STATIC third_party/sqlite/sqlite3.c)
target_include_directories(sb_sqlite PUBLIC third_party/sqlite)
target_compile_definitions(sb_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION SQLITE_ENABLE_JSON1)
target_compile_options(sb_sqlite PRIVATE -w)
target_link_libraries(sb_sqlite PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

set(SB_WARNINGS -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wno-sign-conversion -Wno-maybe-uninitialized)

# Core: schema, writer, blobs, encoders, telemetry ingest helpers.
add_library(sbcore STATIC
  src/core/db.cpp
  src/core/hash.cpp
  src/core/blob_storage.cpp
  src/core/png.cpp
  src/core/wav.cpp
  src/core/session_guard.cpp
  src/core/summary_writer.cpp
  src/core/system_metrics.cpp
  src/core/c_api.cpp)
target_include_directories(sbcore PUBLIC include)
target_include_directories(sbcore SYSTEM PUBLIC third_party)
target_link_libraries(sbcore PUBLIC sb_sqlite ZLIB::ZLIB Threads::Threads)
target_compile_options(sbcore PRIVATE ${SB_WARNINGS})

# Server: HTTP/1.1 + WebSocket, data provider, run manager, live updates, routes.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/server/app.cpp)
add_library(sbserver STATIC
  src/server/http.cpp
  src/server/data_provider.cpp
  src/server/run_manager.cpp
  src/server/live_updates.cpp
  src/server/routes.cpp
  src/server/app.cpp)
target_include_directories(sbserver PUBLIC include)
target_include_directories(sbserver SYSTEM PUBLIC third_party)
target_link_libraries(sbserver PUBLIC sbcore)
target_compile_options(sbserver PRIVATE ${SB_WARNINGS})

add_executable(serenityboardd tools/serenityboardd.cpp)
target_link_libraries(serenityboardd PRIVATE sbserver)
target_compile_options(serenityboardd PRIVATE ${SB_WARNINGS})
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/server_tests.cpp)
add_executable(sb_server_tests tests/server_tests.cpp)
target_link_libraries(sb_server_tests PRIVATE sbserver)
target_compile_options(sb_server_tests PRIVATE ${SB_WARNINGS})
add_test(NAME server COMMAND sb_server_tests)
endif()
endif()

# Ingest: Diffusion Compiler telemetry / reports / step lines -> board.db
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/ingest/compiler_ingest.cpp)
add_executable(sbingest tools/sbingest.cpp src/ingest/compiler_ingest.cpp)
target_link_libraries(sbingest PRIVATE sbcore)
target_compile_options(sbingest PRIVATE ${SB_WARNINGS})
endif()

# Tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/core_tests.cpp)
add_executable(sb_core_tests tests/core_tests.cpp)
target_link_libraries(sb_core_tests PRIVATE sbcore)
target_compile_options(sb_core_tests PRIVATE ${SB_WARNINGS})
add_test(NAME core COMMAND sb_core_tests)
endif()
