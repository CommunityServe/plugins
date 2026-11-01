// Profile builds only (`zig build -Dplugin-profile=true`): the command buffer the verify timeline's shim hooks
// (dsv41_cb_timeline.mm), in place of mlx-serve's dsv41_tl_mlx.cpp, which reads MLX's current buffer through MLX's
// private headers. A buffer of this plugin's own queue on the system device has the class chain of MLX's buffers (the
// device's), so hooking its chain hooks theirs; the shim takes the first recorded buffer's queue as MLX's
// (patches/turboexl3-zzz2-verify-timeline.patch). One buffer a process, never committed. Non-ARC Objective-C.
#import <Metal/Metal.h>

#include "dsv41_cb_timeline.h"

void *dsv41tl_mlx_buffer(mlx_stream s) {
  (void)s;
  static id<MTLCommandBuffer> buf = nil;
  if (buf == nil) {
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> q = dev == nil ? nil : [dev newCommandQueue];
      if (q != nil) buf = [[q commandBufferWithUnretainedReferences] retain];
    }
  }
  return buf;
}
