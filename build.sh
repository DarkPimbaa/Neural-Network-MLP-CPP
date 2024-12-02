#!/bin/bash

# Criar e entrar no diretório de build
mkdir -p build
cd build

# Configurar o CMake
cmake ..

# Compilar
make

# Copiar as bibliotecas para a pasta lib
mkdir -p ../lib
cp lib/libredeneural.* ../lib/

echo "Bibliotecas geradas em ./lib:"
ls -l ../lib
