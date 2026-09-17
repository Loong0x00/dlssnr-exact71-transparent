#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <variant>
#include <vector>

using NgxResult = std::uint32_t;
using NgxFeature = std::int32_t;
struct NgxHandle;

constexpr NgxResult kSuccess = 1;
constexpr NgxResult kInvalidParameter = 0xbad00005u;
constexpr NgxFeature kDlssNrFeature = 18;

struct Parameter {
    virtual void SetULL(const char *, unsigned long long) = 0;            // 0
    virtual void SetFloat(const char *, float) = 0;                       // 1
    virtual void SetDouble(const char *, double) = 0;                     // 2
    virtual void SetUInt(const char *, unsigned int) = 0;                 // 3
    virtual void SetInt(const char *, int) = 0;                           // 4
    virtual void SetD3D11(const char *, void *) = 0;                      // 5
    virtual void SetD3D12(const char *, const void *) = 0;                // 6
    virtual void SetPointer(const char *, volatile void *) = 0;           // 7
    virtual NgxResult GetULL(const char *, unsigned long long *) const = 0;// 8
    virtual NgxResult GetFloat(const char *, float *) const = 0;          // 9
    virtual NgxResult GetDouble(const char *, double *) const = 0;        // 10
    virtual NgxResult GetUInt(const char *, unsigned int *) const = 0;    // 11
    virtual NgxResult GetInt(const char *, int *) const = 0;              // 12
    virtual NgxResult GetD3D11(const char *, void **) const = 0;          // 13
    // DLSSNR 310.8's private CUDA parameter ABI uses slot 14 for float
    // reads, even though the public NGX header assigns that slot to D3D12.
    virtual NgxResult GetPrivateFloat(const char *, float *) const = 0;  // 14
    virtual NgxResult GetPointer(const char *, volatile void **) const = 0;// 15
    virtual void Reset() = 0;
};

class TraceParameters final : public Parameter {
public:
    using Value = std::variant<unsigned long long, float, double, unsigned int, int, void *>;

    void Put(const char *name, unsigned long long value) { values_[name] = value; }
    void Put(const char *name, unsigned int value) { values_[name] = value; }
    void Put(const char *name, int value) { values_[name] = value; }
    void Put(const char *name, float value) { values_[name] = value; }
    void Put(const char *name, void *value) { values_[name] = value; }

    void SetULL(const char *n, unsigned long long v) override { put(n, Value(v), "u64"); }
    void SetFloat(const char *n, float v) override { put(n, Value(v), "f32"); }
    void SetDouble(const char *n, double v) override { put(n, Value(v), "f64"); }
    void SetUInt(const char *n, unsigned int v) override { put(n, Value(v), "u32"); }
    void SetInt(const char *n, int v) override { put(n, Value(v), "i32"); }
    void SetD3D11(const char *n, void *v) override { put(n, Value(v), "d3d11"); }
    void SetD3D12(const char *n, const void *v) override { put(n, Value(const_cast<void *>(v)), "d3d12"); }
    void SetPointer(const char *n, volatile void *v) override {
        put(n, Value(const_cast<void *>(v)), "ptr");
    }
    NgxResult GetULL(const char *n, unsigned long long *v) const override { return get(n, v, "u64"); }
    NgxResult GetFloat(const char *n, float *v) const override { return get(n, v, "f32"); }
    NgxResult GetDouble(const char *n, double *v) const override { return get(n, v, "f64"); }
    NgxResult GetUInt(const char *n, unsigned int *v) const override { return get(n, v, "u32"); }
    NgxResult GetInt(const char *n, int *v) const override { return get(n, v, "i32"); }
    NgxResult GetD3D11(const char *n, void **v) const override { return get(n, v, "d3d11"); }
    NgxResult GetPrivateFloat(const char *n, float *v) const override {
        return get(n, v, "f32-private");
    }
    NgxResult GetPointer(const char *n, volatile void **v) const override {
        void *temporary = nullptr;
        const NgxResult result = get(n, &temporary, "ptr");
        *v = temporary;
        return result;
    }
    void Reset() override { values_.clear(); }

private:
    std::map<std::string, Value> values_;

    void put(const char *name, Value value, const char *kind) {
        std::printf("SET %-6s %s\n", kind, name ? name : "(null)");
        values_[name ? name : ""] = value;
    }

    template <typename T>
    NgxResult get(const char *name, T *output, const char *kind) const {
        const auto found = values_.find(name ? name : "");
        if (found == values_.end()) {
            std::printf("GET %-6s %-48s -> missing\n", kind, name ? name : "(null)");
            return kInvalidParameter;
        }
        if (const auto *value = std::get_if<T>(&found->second)) {
            *output = *value;
            std::printf("GET %-6s %-48s -> present\n", kind, name ? name : "(null)");
            return kSuccess;
        }
        std::printf("GET %-6s %-48s -> wrong-type\n", kind, name ? name : "(null)");
        return kInvalidParameter;
    }

};

using CUdevice = int;
using CUcontext = void *;
using CUstream = void *;
using CUarray = void *;
using CUtexObject = unsigned long long;
using CUsurfObject = unsigned long long;
using CUevent = void *;
using CUgreenCtx = void *;
using CUdevResourceDesc = void *;
using CUdeviceptr = unsigned long long;
using CUresult = int;
struct CudaDevice { void *context; void *stream; };

enum CUarray_format { CU_AD_FORMAT_FLOAT = 0x20 };
enum CUmemorytype { CU_MEMORYTYPE_HOST = 0x01, CU_MEMORYTYPE_ARRAY = 0x03 };
enum CUresourcetype { CU_RESOURCE_TYPE_ARRAY = 0x00 };
enum CUaddress_mode { CU_TR_ADDRESS_MODE_CLAMP = 1 };
enum CUfilter_mode { CU_TR_FILTER_MODE_POINT = 0 };
constexpr unsigned int CUDA_ARRAY3D_SURFACE_LDST = 0x02;
constexpr unsigned int CU_TRSF_NORMALIZED_COORDINATES = 0x02;
constexpr unsigned int CU_STREAM_NON_BLOCKING = 0x01;
constexpr unsigned int CU_GREEN_CTX_DEFAULT_STREAM = 0x01;
constexpr unsigned int CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING = 0x01;

enum CUdevResourceType {
    CU_DEV_RESOURCE_TYPE_INVALID = 0,
    CU_DEV_RESOURCE_TYPE_SM = 1,
};

struct CUdevSmResource {
    unsigned int smCount;
    unsigned int minSmPartitionSize;
    unsigned int smCoscheduledAlignment;
    unsigned int flags;
};

struct CUdevResource {
    CUdevResourceType type;
    unsigned char internal_padding[92];
    union {
        CUdevSmResource sm;
        unsigned char oversize[40];
    } resource;
    CUdevResource *nextResource;
};
static_assert(sizeof(CUdevResource) == 144, "CUDA resource ABI mismatch");

struct CUDA_ARRAY3D_DESCRIPTOR {
    std::size_t Width;
    std::size_t Height;
    std::size_t Depth;
    CUarray_format Format;
    unsigned int NumChannels;
    unsigned int Flags;
};

struct CUDA_RESOURCE_DESC {
    CUresourcetype resType;
    union {
        struct { CUarray hArray; } array;
        struct { int reserved[32]; } reserved;
    } res;
    unsigned int flags;
};

struct CUDA_TEXTURE_DESC {
    CUaddress_mode addressMode[3];
    CUfilter_mode filterMode;
    unsigned int flags;
    unsigned int maxAnisotropy;
    CUfilter_mode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};

struct CUDA_MEMCPY2D {
    std::size_t srcXInBytes;
    std::size_t srcY;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    std::size_t srcPitch;
    std::size_t dstXInBytes;
    std::size_t dstY;
    CUmemorytype dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    std::size_t dstPitch;
    std::size_t WidthInBytes;
    std::size_t Height;
};

using CuInit = CUresult (WINAPI *)(unsigned int);
using CuDeviceGet = CUresult (WINAPI *)(CUdevice *, int);
using CuCtxCreate = CUresult (WINAPI *)(CUcontext *, unsigned int, CUdevice);
using CuCtxSetCurrent = CUresult (WINAPI *)(CUcontext);
using CuCtxSynchronize = CUresult (WINAPI *)();
using CuStreamCreate = CUresult (WINAPI *)(CUstream *, unsigned int);
using CuStreamDestroy = CUresult (WINAPI *)(CUstream);
using CuCtxDestroy = CUresult (WINAPI *)(CUcontext);
using CuArray3DCreate = CUresult (WINAPI *)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
using CuArrayDestroy = CUresult (WINAPI *)(CUarray);
using CuTexObjectCreate = CUresult (WINAPI *)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                                              const CUDA_TEXTURE_DESC *, const void *);
using CuTexObjectDestroy = CUresult (WINAPI *)(CUtexObject);
using CuSurfObjectCreate = CUresult (WINAPI *)(CUsurfObject *, const CUDA_RESOURCE_DESC *);
using CuSurfObjectDestroy = CUresult (WINAPI *)(CUsurfObject);
using CuMemcpy2D = CUresult (WINAPI *)(const CUDA_MEMCPY2D *);
using CuStreamSynchronize = CUresult (WINAPI *)(CUstream);
using CuEventCreate = CUresult (WINAPI *)(CUevent *, unsigned int);
using CuEventRecord = CUresult (WINAPI *)(CUevent, CUstream);
using CuEventSynchronize = CUresult (WINAPI *)(CUevent);
using CuEventElapsedTime = CUresult (WINAPI *)(float *, CUevent, CUevent);
using CuEventDestroy = CUresult (WINAPI *)(CUevent);
using CuDeviceGetDevResource = CUresult (WINAPI *)(CUdevice, CUdevResource *, CUdevResourceType);
using CuDevSmResourceSplitByCount = CUresult (WINAPI *)(CUdevResource *, unsigned int *,
                                                        const CUdevResource *, CUdevResource *,
                                                        unsigned int, unsigned int);
using CuDevResourceGenerateDesc = CUresult (WINAPI *)(CUdevResourceDesc *, CUdevResource *,
                                                      unsigned int);
using CuGreenCtxCreate = CUresult (WINAPI *)(CUgreenCtx *, CUdevResourceDesc, CUdevice,
                                             unsigned int);
using CuCtxFromGreenCtx = CUresult (WINAPI *)(CUcontext *, CUgreenCtx);
using CuGreenCtxStreamCreate = CUresult (WINAPI *)(CUstream *, CUgreenCtx, unsigned int, int);
using CuGreenCtxDestroy = CUresult (WINAPI *)(CUgreenCtx);

using NgxInitExt1 = NgxResult (__cdecl *)(unsigned long long, const wchar_t *, void *,
                                         std::uint32_t, const void *);
using NgxCreateFeature = NgxResult (__cdecl *)(NgxFeature, const Parameter *, NgxHandle **);
using NgxCreateFeature1 = NgxResult (__cdecl *)(CudaDevice *, NgxFeature, const Parameter *, NgxHandle **);
using NgxEvaluateFeature = NgxResult (__cdecl *)(const NgxHandle *, const Parameter *, void *);
using NgxReleaseFeature = NgxResult (__cdecl *)(NgxHandle *);
using NgxShutdown1 = NgxResult (__cdecl *)(CudaDevice *);
using NgxGetApiVersion = std::uint32_t (__cdecl *)();

static HMODULE g_spoof_module = nullptr;
static DWORD (WINAPI *g_original_get_module_filename_w)(HMODULE, LPWSTR, DWORD) = nullptr;

static DWORD WINAPI spoof_get_module_filename_w(HMODULE module, LPWSTR filename, DWORD size) {
    if (module == g_spoof_module && filename && size) {
        static constexpr wchar_t fake[] = L"nvngx.dll";
        constexpr DWORD length = static_cast<DWORD>(sizeof(fake) / sizeof(fake[0]) - 1);
        if (size <= length)
            return 0;
        std::memcpy(filename, fake, sizeof(fake));
        return length;
    }
    return g_original_get_module_filename_w
        ? g_original_get_module_filename_w(module, filename, size) : 0;
}

static bool patch_import(HMODULE module, const char *name, void *replacement, void **previous) {
    auto *base = reinterpret_cast<std::uint8_t *>(module);
    auto *dos = reinterpret_cast<IMAGE_DOS_HEADER *>(base);
    if (!dos || dos->e_magic != IMAGE_DOS_SIGNATURE)
        return false;
    auto *nt = reinterpret_cast<IMAGE_NT_HEADERS64 *>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE)
        return false;
    const auto &directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    auto *descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + directory.VirtualAddress);
    for (; descriptor->Name; ++descriptor) {
        auto *lookup = reinterpret_cast<IMAGE_THUNK_DATA64 *>(base +
            (descriptor->OriginalFirstThunk ? descriptor->OriginalFirstThunk : descriptor->FirstThunk));
        auto *iat = reinterpret_cast<IMAGE_THUNK_DATA64 *>(base + descriptor->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal))
                continue;
            auto *entry = reinterpret_cast<IMAGE_IMPORT_BY_NAME *>(base + lookup->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char *>(entry->Name), name) != 0)
                continue;
            DWORD old = 0;
            if (!VirtualProtect(&iat->u1.Function, sizeof(iat->u1.Function), PAGE_READWRITE, &old))
                return false;
            *previous = reinterpret_cast<void *>(iat->u1.Function);
            iat->u1.Function = reinterpret_cast<ULONGLONG>(replacement);
            DWORD ignored = 0;
            VirtualProtect(&iat->u1.Function, sizeof(iat->u1.Function), old, &ignored);
            return true;
        }
    }
    return false;
}

static bool patch_cuda_context_private_call(HMODULE module) {
    // CUDA CreateFeature1 at RVA 0x265c0 calls a Windows-driver-private
    // CUcontext method at context+0x2c0. A CUcontext is intentionally opaque;
    // Proton's relay cannot and should not reproduce that private layout.
    // cuCtxCreate_v2 has already made this context current. Keep the call in
    // place in case it has harmless bookkeeping side effects, but ignore its
    // Windows-private layout validation result and take the success branch.
    auto *site = reinterpret_cast<std::uint8_t *>(module) + 0x26773;
    static constexpr std::uint8_t jump_to_success[] = {0xeb, 0x07};
    DWORD old = 0;
    if (!VirtualProtect(site, sizeof(jump_to_success), PAGE_EXECUTE_READWRITE, &old))
        return false;
    std::memcpy(site, jump_to_success, sizeof(jump_to_success));
    FlushInstructionCache(GetCurrentProcess(), site, sizeof(jump_to_success));
    DWORD ignored = 0;
    VirtualProtect(site, sizeof(jump_to_success), old, &ignored);
    return true;
}

static bool patch_cuda_architecture_query(HMODULE module) {
    // The snippet asks a Windows-private CUDA object for the NGX architecture
    // code. Linux exposes the same physical RTX 5090 through the public CUDA
    // API but not that object ABI. Return the verified Blackwell code (0x1b0).
    auto *site = reinterpret_cast<std::uint8_t *>(module) + 0x17e94;
    static constexpr std::uint8_t blackwell_result[] = {
        0xb8, 0xb0, 0x01, 0x00, 0x00, 0x90
    };
    DWORD old = 0;
    if (!VirtualProtect(site, sizeof(blackwell_result), PAGE_EXECUTE_READWRITE, &old))
        return false;
    std::memcpy(site, blackwell_result, sizeof(blackwell_result));
    FlushInstructionCache(GetCurrentProcess(), site, sizeof(blackwell_result));
    DWORD ignored = 0;
    VirtualProtect(site, sizeof(blackwell_result), old, &ignored);
    return true;
}

template <typename T>
T symbol(HMODULE module, const char *name) {
    auto pointer = reinterpret_cast<T>(GetProcAddress(module, name));
    std::printf("symbol %-40s %p\n", name, reinterpret_cast<void *>(pointer));
    return pointer;
}

struct CudaImage {
    CUarray array = nullptr;
    CUtexObject texture = 0;
    CUsurfObject surface = 0;
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int channels = 0;
};

static bool create_cuda_image(CudaImage &image, unsigned int width, unsigned int height,
                              unsigned int channels, bool make_texture, bool make_surface,
                              CuArray3DCreate array_create, CuTexObjectCreate texture_create,
                              CuSurfObjectCreate surface_create) {
    image.width = width;
    image.height = height;
    image.channels = channels;
    CUDA_ARRAY3D_DESCRIPTOR array_desc{};
    array_desc.Width = width;
    array_desc.Height = height;
    array_desc.Depth = 0;
    array_desc.Format = CU_AD_FORMAT_FLOAT;
    array_desc.NumChannels = channels;
    array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
    CUresult result = array_create(&image.array, &array_desc);
    std::printf("cuArray3DCreate %ux%ux%u = %d array=%p\n",
                width, height, channels, result, image.array);
    if (result != 0)
        return false;

    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = image.array;
    if (make_texture) {
        CUDA_TEXTURE_DESC texture_desc{};
        texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        texture_desc.filterMode = CU_TR_FILTER_MODE_POINT;
        texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
        texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        result = texture_create(&image.texture, &resource, &texture_desc, nullptr);
        std::printf("cuTexObjectCreate = %d texture=0x%llx\n", result, image.texture);
        if (result != 0)
            return false;
    }
    if (make_surface) {
        result = surface_create(&image.surface, &resource);
        std::printf("cuSurfObjectCreate = %d surface=0x%llx\n", result, image.surface);
        if (result != 0)
            return false;
    }
    return true;
}

static bool copy_host_to_array(const CudaImage &image, const float *pixels, CuMemcpy2D copy) {
    CUDA_MEMCPY2D descriptor{};
    descriptor.srcMemoryType = CU_MEMORYTYPE_HOST;
    descriptor.srcHost = pixels;
    descriptor.srcPitch = static_cast<std::size_t>(image.width) * image.channels * sizeof(float);
    descriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    descriptor.dstArray = image.array;
    descriptor.WidthInBytes = descriptor.srcPitch;
    descriptor.Height = image.height;
    const CUresult result = copy(&descriptor);
    std::printf("cuMemcpy2D host->array %ux%ux%u = %d\n",
                image.width, image.height, image.channels, result);
    return result == 0;
}

static bool copy_array_to_host(const CudaImage &image, float *pixels, CuMemcpy2D copy) {
    CUDA_MEMCPY2D descriptor{};
    descriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    descriptor.srcArray = image.array;
    descriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
    descriptor.dstHost = pixels;
    descriptor.dstPitch = static_cast<std::size_t>(image.width) * image.channels * sizeof(float);
    descriptor.WidthInBytes = descriptor.dstPitch;
    descriptor.Height = image.height;
    const CUresult result = copy(&descriptor);
    std::printf("cuMemcpy2D array->host %ux%ux%u = %d\n",
                image.width, image.height, image.channels, result);
    return result == 0;
}

static void destroy_cuda_image(CudaImage &image, CuTexObjectDestroy texture_destroy,
                               CuSurfObjectDestroy surface_destroy, CuArrayDestroy array_destroy) {
    if (image.texture && texture_destroy)
        std::printf("cuTexObjectDestroy=0x%x\n", texture_destroy(image.texture));
    if (image.surface && surface_destroy)
        std::printf("cuSurfObjectDestroy=0x%x\n", surface_destroy(image.surface));
    if (image.array && array_destroy)
        std::printf("cuArrayDestroy=0x%x\n", array_destroy(image.array));
    image = {};
}

static bool load_float_file(const char *path, std::vector<float> &pixels) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return false;
    input.read(reinterpret_cast<char *>(pixels.data()),
               static_cast<std::streamsize>(pixels.size() * sizeof(float)));
    return input.good() || input.gcount() == static_cast<std::streamsize>(pixels.size() * sizeof(float));
}

static bool save_float_file(const char *path, const std::vector<float> &pixels) {
    std::ofstream output(path, std::ios::binary);
    if (!output)
        return false;
    output.write(reinterpret_cast<const char *>(pixels.data()),
                 static_cast<std::streamsize>(pixels.size() * sizeof(float)));
    return output.good();
}

static void dump_cuda_state(HMODULE runtime, CudaDevice *device) {
    using FindState = void *(__fastcall *)(std::uintptr_t);
    auto find_state = reinterpret_cast<FindState>(
        reinterpret_cast<std::uint8_t *>(runtime) + 0x26270);
    auto *state = static_cast<std::uint8_t *>(
        find_state(reinterpret_cast<std::uintptr_t>(device)));
    std::printf("cuda_state=%p\n", state);
    if (!state)
        return;
    for (std::size_t offset = 0; offset < 0x98; offset += 8) {
        std::uint64_t value = 0;
        std::memcpy(&value, state + offset, sizeof(value));
        std::printf("  state+0x%02zx = 0x%016llx\n", offset,
                    static_cast<unsigned long long>(value));
    }
    void *feature_manager = nullptr;
    std::memcpy(&feature_manager, state + 0x40, sizeof(feature_manager));
    std::printf("feature_manager=%p\n", feature_manager);
    if (feature_manager) {
        const auto *bytes = static_cast<const std::uint8_t *>(feature_manager);
        for (std::size_t offset = 0; offset < 0x40; offset += 8) {
            std::uint64_t value = 0;
            std::memcpy(&value, bytes + offset, sizeof(value));
            std::printf("  manager+0x%02zx = 0x%016llx\n", offset,
                        static_cast<unsigned long long>(value));
        }
    }
}

int main(int argc, char **argv) {
    std::freopen("cuda-probe-inner.log", "w", stdout);
    std::freopen("cuda-probe-error.log", "w", stderr);
    if (argc != 2 && (argc < 6 || argc > 19)) {
        std::fprintf(stderr,
            "usage: cuda_feature_probe.exe nvngx_dlssnr.dll [input.rgba32f output.rgba32f width height"
            " [intensity scaling_ratio tone structure skin style auto_mask depth control_mask iterations"
            " gpc_units sm_count_override render_preset]]\n");
        return 2;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const unsigned int width = argc >= 6 ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10)) : 256;
    const unsigned int height = argc >= 6 ? static_cast<unsigned int>(std::strtoul(argv[5], nullptr, 10)) : 256;
    const char *output_path = argc >= 6 ? argv[3] : "cuda-output.rgba32f";
    const float intensity = argc > 6 ? std::strtof(argv[6], nullptr) : 1.0f;
    const float scaling_ratio = argc > 7 ? std::strtof(argv[7], nullptr) : 1.0f;
    const float tone_strength = argc > 8 ? std::strtof(argv[8], nullptr) : 1.0f;
    const float structure_strength = argc > 9 ? std::strtof(argv[9], nullptr) : 1.0f;
    const float skin_strength = argc > 10 ? std::strtof(argv[10], nullptr) : -1.0f;
    const int style = argc > 11 ? std::strtol(argv[11], nullptr, 10) : 0;
    const unsigned int auto_mask = argc > 12 ? std::strtoul(argv[12], nullptr, 10) : 0u;
    const float depth_value = argc > 13 ? std::strtof(argv[13], nullptr) : 0.5f;
    const float control_mask_value = argc > 14 ? std::strtof(argv[14], nullptr) : -1.0f;
    const unsigned int iterations = argc > 15 ? std::strtoul(argv[15], nullptr, 10) : 1u;
    const unsigned int gpc_units = argc > 16 ? std::strtoul(argv[16], nullptr, 10) : 0u;
    const unsigned int sm_count_override = argc > 17 ? std::strtoul(argv[17], nullptr, 10) : 0u;
    const unsigned int render_preset = argc > 18 ? std::strtoul(argv[18], nullptr, 10) : 0u;
    if (!width || !height || !iterations || gpc_units > 11 || sm_count_override > 170)
        return 2;

    // Avoid a same-directory PE placeholder masking Proton's builtin
    // nvcuda Unix bridge.
    HMODULE cuda = LoadLibraryW(L"C:\\windows\\system32\\nvcuda.dll");
    std::printf("nvcuda=%p error=%lu\n", cuda, GetLastError());
    if (!cuda)
        return 3;
    auto cu_init = symbol<CuInit>(cuda, "cuInit");
    auto cu_device_get = symbol<CuDeviceGet>(cuda, "cuDeviceGet");
    auto cu_ctx_create = symbol<CuCtxCreate>(cuda, "cuCtxCreate_v2");
    auto cu_ctx_set_current = symbol<CuCtxSetCurrent>(cuda, "cuCtxSetCurrent");
    auto cu_ctx_synchronize = symbol<CuCtxSynchronize>(cuda, "cuCtxSynchronize");
    auto cu_stream_create = symbol<CuStreamCreate>(cuda, "cuStreamCreate");
    auto cu_stream_destroy = symbol<CuStreamDestroy>(cuda, "cuStreamDestroy_v2");
    auto cu_ctx_destroy = symbol<CuCtxDestroy>(cuda, "cuCtxDestroy_v2");
    auto cu_array_create = symbol<CuArray3DCreate>(cuda, "cuArray3DCreate_v2");
    auto cu_array_destroy = symbol<CuArrayDestroy>(cuda, "cuArrayDestroy");
    auto cu_texture_create = symbol<CuTexObjectCreate>(cuda, "cuTexObjectCreate");
    auto cu_texture_destroy = symbol<CuTexObjectDestroy>(cuda, "cuTexObjectDestroy");
    auto cu_surface_create = symbol<CuSurfObjectCreate>(cuda, "cuSurfObjectCreate");
    auto cu_surface_destroy = symbol<CuSurfObjectDestroy>(cuda, "cuSurfObjectDestroy");
    auto cu_memcpy_2d = symbol<CuMemcpy2D>(cuda, "cuMemcpy2D_v2");
    auto cu_stream_synchronize = symbol<CuStreamSynchronize>(cuda, "cuStreamSynchronize");
    auto cu_event_create = symbol<CuEventCreate>(cuda, "cuEventCreate");
    auto cu_event_record = symbol<CuEventRecord>(cuda, "cuEventRecord");
    auto cu_event_synchronize = symbol<CuEventSynchronize>(cuda, "cuEventSynchronize");
    auto cu_event_elapsed = symbol<CuEventElapsedTime>(cuda, "cuEventElapsedTime");
    auto cu_event_destroy = symbol<CuEventDestroy>(cuda, "cuEventDestroy_v2");
    auto cu_device_get_resource = symbol<CuDeviceGetDevResource>(cuda, "cuDeviceGetDevResource");
    auto cu_sm_split = symbol<CuDevSmResourceSplitByCount>(cuda, "cuDevSmResourceSplitByCount");
    auto cu_resource_desc = symbol<CuDevResourceGenerateDesc>(cuda, "cuDevResourceGenerateDesc");
    auto cu_green_create = symbol<CuGreenCtxCreate>(cuda, "cuGreenCtxCreate");
    auto cu_ctx_from_green = symbol<CuCtxFromGreenCtx>(cuda, "cuCtxFromGreenCtx");
    auto cu_green_stream_create = symbol<CuGreenCtxStreamCreate>(cuda, "cuGreenCtxStreamCreate");
    auto cu_green_destroy = symbol<CuGreenCtxDestroy>(cuda, "cuGreenCtxDestroy");
    if (!cu_init || !cu_device_get || !cu_ctx_create || !cu_stream_create || !cu_ctx_synchronize)
        return 4;
    if (!cu_array_create || !cu_array_destroy || !cu_texture_create || !cu_texture_destroy ||
        !cu_surface_create || !cu_surface_destroy || !cu_memcpy_2d || !cu_stream_synchronize)
        return 4;
    if (!cu_event_create || !cu_event_record || !cu_event_synchronize ||
        !cu_event_elapsed || !cu_event_destroy)
        return 4;

    CUdevice ordinal = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUgreenCtx green_context = nullptr;
    CUresult cuda_result = cu_init(0);
    std::printf("cuInit=%d\n", cuda_result);
    if (cuda_result != 0)
        return 5;
    cuda_result = cu_device_get(&ordinal, 0);
    std::printf("cuDeviceGet=%d device=%d\n", cuda_result, ordinal);
    if (cuda_result != 0)
        return 6;
    if (gpc_units || sm_count_override) {
        if (!cu_ctx_set_current || !cu_device_get_resource || !cu_sm_split ||
            !cu_resource_desc || !cu_green_create || !cu_ctx_from_green ||
            !cu_green_stream_create || !cu_green_destroy) {
            std::printf("green_context_symbols_missing=1\n");
            return 8;
        }
        CUdevResource full_resource{};
        CUdevResource selected_resource{};
        CUdevResource remainder{};
        cuda_result = cu_device_get_resource(ordinal, &full_resource, CU_DEV_RESOURCE_TYPE_SM);
        std::printf("cuDeviceGetDevResource=%d total_sm=%u min_partition=%u alignment=%u\n",
                    cuda_result, full_resource.resource.sm.smCount,
                    full_resource.resource.sm.minSmPartitionSize,
                    full_resource.resource.sm.smCoscheduledAlignment);
        if (cuda_result != 0)
            return 8;
        const unsigned int requested_sms = sm_count_override ? sm_count_override :
            (gpc_units == 11 ? full_resource.resource.sm.smCount : gpc_units * 16);
        if (requested_sms == full_resource.resource.sm.smCount) {
            selected_resource = full_resource;
        } else {
            unsigned int group_count = 1;
            cuda_result = cu_sm_split(&selected_resource, &group_count, &full_resource,
                                      &remainder,
                                      sm_count_override ?
                                          CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING : 0,
                                      requested_sms);
            std::printf("cuDevSmResourceSplitByCount=%d requested_sm=%u groups=%u "
                        "selected_sm=%u remainder_sm=%u\n",
                        cuda_result, requested_sms, group_count,
                        selected_resource.resource.sm.smCount,
                        remainder.resource.sm.smCount);
            if (cuda_result != 0 || group_count != 1)
                return 8;
        }
        CUdevResourceDesc descriptor = nullptr;
        cuda_result = cu_resource_desc(&descriptor, &selected_resource, 1);
        std::printf("cuDevResourceGenerateDesc=%d descriptor=%p selected_sm=%u gpc_units=%u "
                    "sm_count_override=%u\n",
                    cuda_result, descriptor, selected_resource.resource.sm.smCount,
                    gpc_units, sm_count_override);
        if (cuda_result != 0)
            return 8;
        cuda_result = cu_green_create(&green_context, descriptor, ordinal,
                                      CU_GREEN_CTX_DEFAULT_STREAM);
        std::printf("cuGreenCtxCreate=%d green=%p\n", cuda_result, green_context);
        if (cuda_result != 0)
            return 8;
        cuda_result = cu_ctx_from_green(&context, green_context);
        std::printf("cuCtxFromGreenCtx=%d context=%p\n", cuda_result, context);
        if (cuda_result != 0)
            return 8;
        cuda_result = cu_ctx_set_current(context);
        std::printf("cuCtxSetCurrent=%d\n", cuda_result);
        if (cuda_result != 0)
            return 8;
        cuda_result = cu_green_stream_create(&stream, green_context, CU_STREAM_NON_BLOCKING, 0);
        std::printf("cuGreenCtxStreamCreate=%d stream=%p\n", cuda_result, stream);
        if (cuda_result != 0)
            return 8;
    } else {
        cuda_result = cu_ctx_create(&context, 0, ordinal);
        std::printf("cuCtxCreate=%d context=%p\n", cuda_result, context);
        if (cuda_result != 0)
            return 7;
        cuda_result = cu_stream_create(&stream, 0);
        std::printf("cuStreamCreate=%d stream=%p\n", cuda_result, stream);
        if (cuda_result != 0)
            return 8;
    }

    HMODULE runtime = LoadLibraryA(argv[1]);
    std::printf("runtime=%p error=%lu\n", runtime, GetLastError());
    if (!runtime)
        return 9;
    g_spoof_module = GetModuleHandleW(nullptr);
    const bool patched = patch_import(runtime, "GetModuleFileNameW",
        reinterpret_cast<void *>(&spoof_get_module_filename_w),
        reinterpret_cast<void **>(&g_original_get_module_filename_w));
    std::printf("identity_patch=%d self=%p\n", patched ? 1 : 0, g_spoof_module);
    std::printf("cuda_context_private_call_patch=unused-single-instance-entry\n");
    const bool architecture_patch = patch_cuda_architecture_query(runtime);
    std::printf("cuda_architecture_patch=%d value=0x1b0\n", architecture_patch ? 1 : 0);
    auto get_api_version = symbol<NgxGetApiVersion>(runtime, "NVSDK_NGX_GetAPIVersion");
    auto init = symbol<NgxInitExt1>(runtime, "NVSDK_NGX_CUDA_Init_Ext1");
    auto create = symbol<NgxCreateFeature>(runtime, "NVSDK_NGX_CUDA_CreateFeature");
    auto evaluate = symbol<NgxEvaluateFeature>(runtime, "NVSDK_NGX_CUDA_EvaluateFeature");
    auto release = symbol<NgxReleaseFeature>(runtime, "NVSDK_NGX_CUDA_ReleaseFeature");
    auto shutdown = symbol<NgxShutdown1>(runtime, "NVSDK_NGX_CUDA_Shutdown1");
    if (!init || !create)
        return 10;

    CudaDevice device{context, stream};
    const std::uint32_t api_version = get_api_version ? get_api_version() : 0x15;
    NgxResult result = init(0x0876232cULL, L".", &device, api_version, nullptr);
    std::printf("NGX CUDA Init_Ext1=0x%08x api=0x%08x\n", result, api_version);
    if (result != kSuccess)
        return 11;
    dump_cuda_state(runtime, &device);

    std::vector<float> color(static_cast<std::size_t>(width) * height * 4);
    if (argc >= 6) {
        if (!load_float_file(argv[2], color)) {
            std::fprintf(stderr, "cannot read exact-size RGBA32F input: %s\n", argv[2]);
            return 13;
        }
    } else {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                const std::size_t p = (static_cast<std::size_t>(y) * width + x) * 4;
                color[p + 0] = static_cast<float>(x) / (width - 1);
                color[p + 1] = static_cast<float>(y) / (height - 1);
                color[p + 2] = ((x / 16 + y / 16) & 1) ? 0.8f : 0.2f;
                color[p + 3] = 1.0f;
            }
        }
    }
    std::vector<float> motion(static_cast<std::size_t>(width) * height * 2, 0.0f);
    std::vector<float> depth(static_cast<std::size_t>(width) * height, depth_value);
    if (depth_value < 0.0f) {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                float value = 0.5f;
                if (depth_value > -1.5f)
                    value = static_cast<float>(x) / (width - 1);
                else if (depth_value > -2.5f)
                    value = static_cast<float>(y) / (height - 1);
                else {
                    const float nx = 2.0f * x / (width - 1) - 1.0f;
                    const float ny = 2.0f * y / (height - 1) - 1.0f;
                    value = (nx * nx + ny * ny) < 0.25f ? 0.1f : 0.9f;
                }
                depth[static_cast<std::size_t>(y) * width + x] = value;
            }
        }
    }
    std::vector<float> control_mask(static_cast<std::size_t>(width) * height,
                                    control_mask_value < 0.0f ? 0.0f : control_mask_value);
    std::vector<float> output(static_cast<std::size_t>(width) * height * 4, 0.0f);

    CudaImage color_image, motion_image, depth_image, output_image, control_mask_image;
    bool resources_ok =
        create_cuda_image(color_image, width, height, 4, true, false,
                          cu_array_create, cu_texture_create, cu_surface_create) &&
        create_cuda_image(motion_image, width, height, 2, true, false,
                          cu_array_create, cu_texture_create, cu_surface_create) &&
        create_cuda_image(depth_image, width, height, 1, true, false,
                          cu_array_create, cu_texture_create, cu_surface_create) &&
        create_cuda_image(output_image, width, height, 4, true, false,
                          cu_array_create, cu_texture_create, cu_surface_create) &&
        copy_host_to_array(color_image, color.data(), cu_memcpy_2d) &&
        copy_host_to_array(motion_image, motion.data(), cu_memcpy_2d) &&
        copy_host_to_array(depth_image, depth.data(), cu_memcpy_2d) &&
        copy_host_to_array(output_image, output.data(), cu_memcpy_2d);
    if (resources_ok && control_mask_value >= 0.0f) {
        resources_ok = create_cuda_image(control_mask_image, width, height, 1, true, false,
                                         cu_array_create, cu_texture_create, cu_surface_create) &&
                       copy_host_to_array(control_mask_image, control_mask.data(), cu_memcpy_2d);
    }
    std::printf("cuda_resources_ok=%d\n", resources_ok ? 1 : 0);
    if (!resources_ok)
        return 14;

    TraceParameters parameters;
    parameters.Put("DLSSNR.Width", static_cast<int>(width));
    parameters.Put("DLSSNR.Height", static_cast<int>(height));
    parameters.Put("DLSSNR.InputWidth", static_cast<int>(width));
    parameters.Put("DLSSNR.InputHeight", static_cast<int>(height));
    parameters.Put("DLSSNR.ScalingRatio", scaling_ratio);
    parameters.Put("DLSSNR.Scale", 1.0f);
    parameters.Put("DLSSNR.Upscaling", 0);
    parameters.Put("DLSSNR.Hint.Render.Preset", render_preset);
    std::printf("DLSSNR.Hint.Render.Preset=%u\n", render_preset);
    parameters.Put("Width", static_cast<int>(width));
    parameters.Put("Height", static_cast<int>(height));
    parameters.Put("OutWidth", static_cast<int>(width));
    parameters.Put("OutHeight", static_cast<int>(height));
    parameters.Put("Input1", context);
    parameters.Put("Input2", stream);
    // NGX's CUDA helper API stores CUtexObject pointers, not CUtexObject values,
    // in the ULL parameter slot. The DLL dereferences these before inspecting
    // and rebinding the underlying CUDA arrays.
    parameters.Put("DLSSNR.Color",
        reinterpret_cast<unsigned long long>(&color_image.texture));
    parameters.Put("DLSSNR.MVec",
        reinterpret_cast<unsigned long long>(&motion_image.texture));
    parameters.Put("DLSSNR.Depth",
        reinterpret_cast<unsigned long long>(&depth_image.texture));
    parameters.Put("DLSSNR.Output",
        reinterpret_cast<unsigned long long>(&output_image.texture));
    if (control_mask_image.texture) {
        parameters.Put("DLSSNR.ControlMask",
            reinterpret_cast<unsigned long long>(&control_mask_image.texture));
    }
    parameters.Put("DLSSNR.MVecScaleX", 1.0f);
    parameters.Put("DLSSNR.MVecScaleY", 1.0f);
    parameters.Put("DLSSNR.Intensity", intensity);
    parameters.Put("DLSSNR.LocalToneStrength", tone_strength);
    parameters.Put("DLSSNR.LocalStructureStrength", structure_strength);
    parameters.Put("DLSSNR.SkinStructureStrength", skin_strength);
    parameters.Put("DLSSNR.UseAutoMask", auto_mask);
    parameters.Put("DLSSNR.Style", style);
    parameters.Put("DLSSNR.Reset", 1u);
    parameters.Put("DLSSNR.DepthInverted", 0u);
    parameters.Put("DLSSNR.Enabled", 1u);
    parameters.Put("DLSSNR.UICorrection", 0u);

    NgxHandle *handle = nullptr;
    result = create(kDlssNrFeature, &parameters, &handle);
    std::printf("NGX CUDA CreateFeature=0x%08x handle=%p\n", result, handle);
    if (result == kSuccess && handle && evaluate) {
        CUevent event_start = nullptr;
        CUevent event_end = nullptr;
        cu_event_create(&event_start, 0);
        cu_event_create(&event_end, 0);
        LARGE_INTEGER frequency{};
        QueryPerformanceFrequency(&frequency);
        NgxResult evaluate_result = kInvalidParameter;
        CUresult sync_result = 0;
        for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
            parameters.Put("DLSSNR.Reset", iteration == 0 ? 1u : 0u);
            LARGE_INTEGER wall_start{}, submit_end{}, wall_end{};
            QueryPerformanceCounter(&wall_start);
            cu_event_record(event_start, stream);
            evaluate_result = evaluate(handle, &parameters, nullptr);
            QueryPerformanceCounter(&submit_end);
            cu_event_record(event_end, stream);
            const CUresult event_sync_result = cu_event_synchronize(event_end);
            sync_result = cu_stream_synchronize(stream);
            const CUresult context_sync_result = cu_ctx_synchronize();
            QueryPerformanceCounter(&wall_end);
            float gpu_milliseconds = -1.0f;
            const CUresult elapsed_result = cu_event_elapsed(
                &gpu_milliseconds, event_start, event_end);
            const double submit_milliseconds = 1000.0 *
                static_cast<double>(submit_end.QuadPart - wall_start.QuadPart) /
                static_cast<double>(frequency.QuadPart);
            const double wall_milliseconds = 1000.0 *
                static_cast<double>(wall_end.QuadPart - wall_start.QuadPart) /
                static_cast<double>(frequency.QuadPart);
            std::printf("TIMING iteration=%u reset=%u result=0x%08x submit_ms=%.6f "
                        "gpu_ms=%.6f wall_ms=%.6f event_sync=%d elapsed=%d stream_sync=%d "
                        "context_sync=%d\n",
                        iteration, iteration == 0 ? 1u : 0u, evaluate_result,
                        submit_milliseconds, gpu_milliseconds, wall_milliseconds,
                        event_sync_result, elapsed_result, sync_result, context_sync_result);
            if (evaluate_result != kSuccess || event_sync_result != 0 ||
                elapsed_result != 0 || sync_result != 0 || context_sync_result != 0)
                break;
        }
        std::printf("NGX CUDA EvaluateFeature(real resources)=0x%08x\n", evaluate_result);
        std::printf("cuStreamSynchronize=%d\n", sync_result);
        if (event_end)
            cu_event_destroy(event_end);
        if (event_start)
            cu_event_destroy(event_start);
        if (evaluate_result == kSuccess && sync_result == 0 &&
            copy_array_to_host(output_image, output.data(), cu_memcpy_2d)) {
            std::printf("save_output=%d path=%s\n", save_float_file(output_path, output) ? 1 : 0,
                        output_path);
        }
    }
    if (handle && release)
        std::printf("NGX CUDA ReleaseFeature=0x%08x\n", release(handle));
    if (shutdown)
        std::printf("NGX CUDA Shutdown1=0x%08x\n", shutdown(&device));
    destroy_cuda_image(control_mask_image, cu_texture_destroy, cu_surface_destroy, cu_array_destroy);
    destroy_cuda_image(output_image, cu_texture_destroy, cu_surface_destroy, cu_array_destroy);
    destroy_cuda_image(depth_image, cu_texture_destroy, cu_surface_destroy, cu_array_destroy);
    destroy_cuda_image(motion_image, cu_texture_destroy, cu_surface_destroy, cu_array_destroy);
    destroy_cuda_image(color_image, cu_texture_destroy, cu_surface_destroy, cu_array_destroy);
    if (cu_stream_destroy)
        std::printf("cuStreamDestroy=%d\n", cu_stream_destroy(stream));
    if (green_context && cu_green_destroy)
        std::printf("cuGreenCtxDestroy=%d\n", cu_green_destroy(green_context));
    else if (cu_ctx_destroy)
        std::printf("cuCtxDestroy=%d\n", cu_ctx_destroy(context));
    return result == kSuccess ? 0 : 12;
}
