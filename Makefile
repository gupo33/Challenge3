debug: main.cpp
	mpic++ -o main main.cpp -DDEBUG

release: main.cpp
	mpic++ -o main main.cpp