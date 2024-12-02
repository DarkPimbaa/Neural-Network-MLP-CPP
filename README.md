# Neural Network
Made with C++ without dependencies

## MLP
MLP for generic use. It can be used in any context where the network does not need to return integer or floating point values.
The network returns a vector(also known as an array) of booleans.

## Features
- Bias neurons in each layer (except output)
- ReLU activation function for hidden layers
- Weight modification for training
- Network truncation support
- Available as both static and dynamic libraries

## Building the Libraries
To build the static and dynamic libraries:

```bash
./build.sh
```

This will create both static (`.a`) and dynamic (`.so`) libraries in the `lib` directory.

## Using the Library
### Include in Your Project
1. Copy the header file from `include/redeNeural.hpp` to your project's include path
2. Copy either the static library (`lib/libredeneural.a`) or dynamic library (`lib/libredeneural.so`) to your project's lib path

### Linking with CMake
```cmake
# For static library
target_link_libraries(your_target /path/to/lib/libredeneural.a)

# For dynamic library
target_link_libraries(your_target /path/to/lib/libredeneural.so)
```

### Basic Usage Example
```cpp
#include <redeNeural.hpp>
#include <vector>

int main() {
    // Create a neural network with:
    // - 3 input neurons (plus 1 bias)
    // - 2 hidden layers (each with bias)
    // - 2 output neurons
    RedeNeural rede(3, 2, 2);
    
    // Input values (excluding bias)
    std::vector<double> inputs = {0.5, -0.5, 1.0};
    
    // Get output
    std::vector<bool> resultado = rede.iniciar(inputs);
    
    return 0;
}

## Reason
I made it with the intention of learning and using it in small future projects.

## License
This project is open source and available under the MIT License.
