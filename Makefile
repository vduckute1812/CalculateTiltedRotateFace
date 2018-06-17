build:
	g++ landmark.cpp -o landmark `pkg-config --cflags --libs opencv`

run:
	./landmark

clean:
	rm -rf landmark