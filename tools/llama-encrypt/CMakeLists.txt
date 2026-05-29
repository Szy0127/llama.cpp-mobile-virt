find_package(OpenSSL REQUIRED)

set(TARGET llama-encrypt)
add_executable(${TARGET} llama-encrypt.cpp)
install(TARGETS ${TARGET} RUNTIME)
# This preprocessing tool only needs gguf primitives and OpenSSL, so it should stay
# buildable on host even when target-only backends are enabled.
target_link_libraries(${TARGET} PRIVATE ggml-base OpenSSL::Crypto ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
