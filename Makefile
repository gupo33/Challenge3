debug: main.cpp
	g++ -o main main.cpp -DDEBUG

release: main.cpp
	g++ -o main main.cpp