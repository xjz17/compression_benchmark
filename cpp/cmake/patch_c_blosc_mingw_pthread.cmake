# Patched copy of c-blosc for ExternalProject (does not modify the submodule).
# On MinGW, blosc.c includes win32/pthread.c which conflicts with libpthread when
# linked into benchmark_cpp alongside other pthread users (e.g. pfpc).
if(NOT src)
  message(FATAL_ERROR "patch_c_blosc_mingw_pthread.cmake: set -Dsrc= to blosc.c path")
endif()
file(READ "${src}" _content)
string(
    REPLACE
    "#if defined(_WIN32)\n  #include \"win32/pthread.h\"\n  #include \"win32/pthread.c\"\n#else\n  #include <pthread.h>\n#endif"
    "#if defined(_WIN32) && defined(__MINGW32__)\n  #include <pthread.h>\n#elif defined(_WIN32)\n  #include \"win32/pthread.h\"\n  #include \"win32/pthread.c\"\n#else\n  #include <pthread.h>\n#endif"
    _content
    "${_content}"
)
file(WRITE "${src}" "${_content}")
