// TURBOEXL3_GLM53F_KEEP_WARM: the C symbols the copied keep-warm engine (src/mlx_serve/sdk/weights/streamed/mimo/
// gpu_warm.zig) calls, as this plugin's own. mlx-serve serves them from csrc/mimo_gpu_warm.inc over MLX's metal-cpp
// device; the host table an outside plugin gets has no command queue of its own, so this file makes the same queue,
// kernel and buffer over Metal's own API: a trivial resident kernel (one dependent FMA chain per thread, the .inc's
// text) on its OWN command queue and its own tiny buffer, committed in short chunks by the engine's warm thread while
// the stream's MTLSharedEvent is below the value its gated waits need. Nothing here reads or writes a model array or
// touches MLX's queue: the kernel's only memory is its own buffer, and the event is only read. The symbols are hidden:
// the engine's references bind to these at the dylib's link, never to the server's own copies.
//
// The device is the system default (MTLCreateSystemDefaultDevice), the one GPU MLX's device is on an Apple-silicon
// Mac. ARC; every call drains its own autorelease pool (the warm thread has none).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define GLM53F_WARM_HIDDEN __attribute__((visibility("hidden")))

static _Thread_local char g_error[512];

// The .inc's kernel, unchanged.
static NSString* const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void mimo_gpu_warm(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],\n"
     "                          uint tid [[thread_position_in_grid]]) {\n"
     "  float a = 1.0f + float(tid) * 1e-7f;\n"
     "  for (uint i = 0; i < iters; i++) {\n"
     "    a = fma(a, 0.9999999f, 1e-7f);\n"
     "  }\n"
     "  out[tid] = a;\n"
     "}\n";

enum { kMaxGroups = 256, kMaxThreads = 256 };

@interface Glm53fWarm : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> pso;
@property(nonatomic, strong) id<MTLBuffer> buf;
@property(nonatomic) uint32_t groups;
@property(nonatomic) uint32_t threads;
@end

@implementation Glm53fWarm
@end

static int fail(const char* what) {
  snprintf(g_error, sizeof g_error, "%s", what);
  return -1;
}

static int fail_ns(const char* what, NSError* err) {
  if (err != nil && err.localizedDescription != nil)
    snprintf(g_error, sizeof g_error, "%s: %s", what, err.localizedDescription.UTF8String);
  else
    snprintf(g_error, sizeof g_error, "%s", what);
  return -1;
}

GLM53F_WARM_HIDDEN const char* mimo_warm_last_error(void) {
  return g_error;
}

// The warm queue, pipeline and buffer on the GPU (`groups` threadgroups of `threads` threads a chunk). Construction
// only (it compiles one tiny library; no kernel runs). 0, or -1.
GLM53F_WARM_HIDDEN int mimo_warm_create(uint32_t groups, uint32_t threads, void** out) {
  *out = NULL;
  if (groups == 0 || groups > kMaxGroups || threads == 0 || threads > kMaxThreads || threads % 32 != 0)
    return fail("groups 1..256 and threads a multiple of 32 up to 256");
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) return fail("no Metal device");
    Glm53fWarm* w = [Glm53fWarm new];
    w.groups = groups;
    w.threads = threads;
    w.queue = [dev newCommandQueue];
    if (w.queue == nil) return fail("newCommandQueue failed");
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
    if (lib == nil) return fail_ns("the warm kernel did not compile", err);
    id<MTLFunction> fn = [lib newFunctionWithName:@"mimo_gpu_warm"];
    if (fn == nil) return fail("the warm kernel's function is missing");
    err = nil;
    w.pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (w.pso == nil) return fail_ns("the warm pipeline did not build", err);
    if (w.pso.maxTotalThreadsPerThreadgroup < threads) return fail("the warm pipeline takes fewer threads per threadgroup than asked");
    w.buf = [dev newBufferWithLength:(NSUInteger)kMaxGroups * kMaxThreads * sizeof(float) options:MTLResourceStorageModeShared];
    if (w.buf == nil) return fail("newBuffer failed");
    *out = (void*)CFBridgingRetain(w);
    return 0;
  }
}

GLM53F_WARM_HIDDEN void mimo_warm_destroy(void* h) {
  if (h == NULL) return;
  @autoreleasepool {
    Glm53fWarm* w = (Glm53fWarm*)CFBridgingRelease(h);
    (void)w;
  }
}

// One chunk: `iters` FMAs a thread, encoded and committed on the warm queue. Writes the retained command buffer to *cb
// (the caller retires it). The warm thread only. 0, or -1.
GLM53F_WARM_HIDDEN int mimo_warm_commit(void* h, uint32_t iters, void** cb) {
  *cb = NULL;
  if (h == NULL) return fail("no warm device");
  @autoreleasepool {
    Glm53fWarm* w = (__bridge Glm53fWarm*)h;
    id<MTLCommandBuffer> buf = [w.queue commandBufferWithUnretainedReferences];
    if (buf == nil) return fail("commandBufferWithUnretainedReferences failed");
    id<MTLComputeCommandEncoder> enc = [buf computeCommandEncoder];
    if (enc == nil) return fail("computeCommandEncoder failed");
    [enc setComputePipelineState:w.pso];
    [enc setBuffer:w.buf offset:0 atIndex:0];
    [enc setBytes:&iters length:sizeof(iters) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(w.groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(w.threads, 1, 1)];
    [enc endEncoding];
    [buf commit];
    *cb = (void*)CFBridgingRetain(buf);
    return 0;
  }
}

// Waits for a committed chunk, writes its GPU start and end (s, the host's mach-absolute clock), releases it. 0, or -1
// when Metal completed it with an error.
GLM53F_WARM_HIDDEN int mimo_warm_retire(void* cb, double* gpu_start, double* gpu_end) {
  *gpu_start = 0;
  *gpu_end = 0;
  if (cb == NULL) return fail("no chunk");
  @autoreleasepool {
    id<MTLCommandBuffer> buf = (id<MTLCommandBuffer>)CFBridgingRelease(cb);
    [buf waitUntilCompleted];
    const MTLCommandBufferStatus st = buf.status;
    *gpu_start = buf.GPUStartTime;
    *gpu_end = buf.GPUEndTime;
    if (st != MTLCommandBufferStatusCompleted) return fail("a warm chunk completed with an error");
    return 0;
  }
}

// The stream's event (an id<MTLSharedEvent>, the host table's `Event.object`): its signaled value.
GLM53F_WARM_HIDDEN uint64_t mimo_warm_event_value(uint64_t event) {
  if (event == 0) return 0;
  id<MTLSharedEvent> ev = (__bridge id<MTLSharedEvent>)(void*)(uintptr_t)event;
  return ev.signaledValue;
}

// The host clock Metal's GPU timestamps use (mach-absolute, ns).
GLM53F_WARM_HIDDEN uint64_t mimo_warm_now_ns(void) {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}
