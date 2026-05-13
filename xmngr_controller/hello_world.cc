#include <iostream>
#include <chrono>
#include <thread>
//g++ -std=c++17 -pthread hello_world.cc -o hello_world
int main() {
    std::cout << "Hello, World!" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    return 0;
}
