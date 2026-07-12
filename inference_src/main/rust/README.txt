the tf proto files were copied from
https://github.com/tensorflow/serving/tree/master/tensorflow_serving
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/framework

on your platform, install the protocol buffere compiler:
    sudo apt install protobuf-compiler
    or use brew install protobuf
    etc...

then, before cargo builds the src code, it invokes build.rs to compile the protocol buffers
