// M3.4b PLE reader self-test (BAS-79).
//
// Proves the C++ reader (O_DIRECT pool + bounded LRU) returns exactly the bytes
// a plain pread returns, including rows that straddle a page boundary and
// repeated/duplicate row ids.  Does not need the model or a GPU.
//
// build (inside the Vulkan build container, from the patched llama.cpp tree):
//   g++ -O2 -std=c++17 ggml/src/ggml-ple-reader.cpp selftest/ple_reader_selftest.cpp \
//       -Iggml/include -Iggml/src -Lbuild-vulkan/bin -lggml-base -lpthread -o /tmp/ple_selftest

#include "ggml-ple-reader.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

static int fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    const uint64_t row_bytes = 90;          // IQ4_NL row for 160 columns
    const int64_t  n_rows    = 200000;
    const uint64_t offset    = 1234;        // deliberately not page aligned
    const uint64_t file_size = offset + (uint64_t) n_rows * row_bytes;
    const std::string path   = "/work/.ple_selftest_data.bin";

    // build a deterministic file
    {
        std::vector<uint8_t> data(file_size);
        uint32_t s = 0x12345678u;
        for (auto & b : data) {
            s = s * 1664525u + 1013904223u;
            b = (uint8_t) (s >> 24);
        }
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) {
            return fail("fopen");
        }
        if (std::fwrite(data.data(), 1, data.size(), f) != data.size()) {
            std::fclose(f);
            return fail("fwrite");
        }
        std::fclose(f);
    }

    // make sure the rows are not page-cached in our reader's fd path: the reader
    // opens its own fd, so this only affects the pread reference below
    const int rfd = ::open(path.c_str(), O_RDONLY);

    // dummy tensor used only as the reader key
    ggml_init_params p = { /*.mem_size =*/ 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, 160, n_rows);

    struct ggml_ple_reader_config cfg = {
        /*.path       =*/ path.c_str(),
        /*.offset     =*/ offset,
        /*.row_bytes  =*/ row_bytes,
        /*.n_rows     =*/ n_rows,
        /*.io_depth   =*/ 16,
        /*.cache_rows =*/ 1000000,
        /*.window     =*/ 256,
    };
    if (!ggml_ple_reader_register(t, &cfg)) {
        return fail("register");
    }
    if (!ggml_ple_reader_has(t)) {
        return fail("has");
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, n_rows - 1);

    std::vector<int32_t> idx(20000);
    for (size_t i = 0; i < idx.size(); ++i) {
        // force duplicates and neighbours so both the cache and page sharing run
        if (i % 7 == 0 && i > 0) {
            idx[i] = idx[i - 1];
        } else if (i % 11 == 0 && i > 0) {
            idx[i] = (int32_t) std::min<int64_t>(n_rows - 1, idx[i - 1] + 1);
        } else {
            idx[i] = (int32_t) dist(rng);
        }
    }

    std::vector<uint8_t> got(idx.size() * row_bytes);
    if (!ggml_ple_reader_gather(t, idx.data(), (int64_t) idx.size(), got.data())) {
        return fail("gather");
    }

    std::vector<uint8_t> want(row_bytes);
    for (size_t i = 0; i < idx.size(); ++i) {
        const uint64_t s = offset + (uint64_t) idx[i] * row_bytes;
        ssize_t n = pread(rfd, want.data(), row_bytes, (off_t) s);
        if (n != (ssize_t) row_bytes) {
            return fail("reference pread");
        }
        if (std::memcmp(want.data(), got.data() + i * row_bytes, row_bytes) != 0) {
            std::fprintf(stderr, "mismatch at request %zu (row %d)\n", i, idx[i]);
            return 1;
        }
    }

    // second pass must be served entirely from the row cache and still be exact
    std::vector<uint8_t> got2(idx.size() * row_bytes);
    if (!ggml_ple_reader_gather(t, idx.data(), (int64_t) idx.size(), got2.data())) {
        return fail("gather2");
    }
    if (std::memcmp(got.data(), got2.data(), got.size()) != 0) {
        return fail("cache pass differs");
    }

    ::close(rfd);
    ggml_ple_reader_clear();
    ggml_free(ctx);
    std::printf("PASS: %zu rows match pread (straddle + duplicates + cache)\n", idx.size());
    return 0;
}
