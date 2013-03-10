find_package(Doxygen)
if(NOT DOXYGEN_FOUND)
    message(FATAL_ERROR
        "Doxygen is needed to build the documentation. Please install it correctly.")
endif()

configure_file(Doxyfile.in
    ${PROJECT_BINARY_DIR}/doc/Doxyfile @ONLY IMMEDIATE)

add_custom_target(docs ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    SOURCES ${PROJECT_BINARY_DIR}/doc/Doxyfile)
