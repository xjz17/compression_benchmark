#pragma once

// Included before the upstream cmix translation units in the isolated Linux
// DSO.  The upstream sources use C allocation spellings in a few places and
// do not consistently check null.  Redirect those calls to ordinary C++
// functions that may throw std::bad_alloc, so the exported codec ABI can catch
// exhaustion and return an error instead of dereferencing null.
// Include every C/C++ system header used by the upstream cmix translation
// units before defining the spelling redirects.  This keeps libc's noexcept
// declarations and std::malloc/std::free imports intact; subsequent upstream
// includes are guarded and only its unqualified call sites are rewritten.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

extern "C" void* native_graph_cmix_checked_malloc(std::size_t size);
extern "C" void* native_graph_cmix_checked_calloc(std::size_t count,
                                                   std::size_t size);
extern "C" void* native_graph_cmix_checked_realloc(void* memory,
                                                    std::size_t size);
extern "C" void native_graph_cmix_checked_free(void* memory) noexcept;
extern "C" void* native_graph_cmix_checked_aligned_alloc(
    std::size_t alignment, std::size_t size);
extern "C" int native_graph_cmix_checked_posix_memalign(
    void** result, std::size_t alignment, std::size_t size);

#define malloc native_graph_cmix_checked_malloc
#define calloc native_graph_cmix_checked_calloc
#define realloc native_graph_cmix_checked_realloc
#define free native_graph_cmix_checked_free
#define aligned_alloc native_graph_cmix_checked_aligned_alloc
#define posix_memalign native_graph_cmix_checked_posix_memalign
