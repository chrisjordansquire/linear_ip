cmake_minimum_required(VERSION 2.8.5)

set(example_SRC
    example.cpp
    )

add_executable(example ${example_SRC})
target_link_libraries(example lp)
