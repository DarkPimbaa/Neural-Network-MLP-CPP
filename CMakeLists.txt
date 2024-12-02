cmake_minimum_required(VERSION 3.10)
project(RedeNeural VERSION 1.0)

# Configurar o C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Definir onde as bibliotecas serão geradas
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Adicionar diretório de include
include_directories(${PROJECT_SOURCE_DIR}/include)

# Criar biblioteca estática
add_library(redeneural_static STATIC
    redeNeural.cpp
)
set_target_properties(redeneural_static PROPERTIES OUTPUT_NAME "redeneural")

# Criar biblioteca dinâmica
add_library(redeneural_shared SHARED
    redeNeural.cpp
)
set_target_properties(redeneural_shared PROPERTIES OUTPUT_NAME "redeneural")

# Criar executável de teste
add_executable(teste_rede main.cpp)
target_link_libraries(teste_rede redeneural_static)

# Instalar bibliotecas e headers
install(TARGETS redeneural_static redeneural_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES include/redeNeural.hpp DESTINATION include)
