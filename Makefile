debug: src/main.cpp include/laplaceSolver.hpp include/writeVTK.hpp
	mpic++ -o main src/main.cpp include/laplaceSolver.hpp include/writeVTK.hpp -Iinclude -Llib -lmuparserx -g -std=c++20 -DDEBUG

release: src/main.cpp include/laplaceSolver.hpp include/writeVTK.hpp
	mpic++ -o main src/main.cpp include/laplaceSolver.hpp include/writeVTK.hpp -Iinclude -Llib -lmuparserx -g -std=c++20 