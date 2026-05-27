set(TARGET llama-rknpu-prepack)
add_executable(${TARGET} rknpu-prepack.cpp)
install(TARGETS ${TARGET} RUNTIME)
# This preprocessing tool only needs ggml/gguf primitives and should stay buildable
# on host even when the RKNPU runtime backend itself is target-only.
target_link_libraries(${TARGET} PRIVATE ggml-base ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../ggml/src/ggml-rknpu_re)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
