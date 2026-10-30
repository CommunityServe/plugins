// normInv: hc_prefill_norm.metal without its normalized output: the same pending write, stream store and reductions, so
// invs[row] is exactly the f32 scale that kernel multiplies by.
const int row=threadgroup_position_in_grid.x;
const int h=row%HC, token=row/HC;
const int tid=thread_index_in_threadgroup, lane=thread_index_in_simdgroup;
const int sg=simdgroup_index_in_threadgroup;
threadgroup float partial[32];
float xv[4], acc=0;
for(int i=0;i<4;i++) {
  T val=x[(long)row*H+tid*4+i];
  if(WR) {
    T delta=T(float(wo[(long)token*H+tid*4+i])*float(wi[token*HC+h]));
    val=T(float(val)+float(delta));
    stream[(long)row*H+tid*4+i]=val;
  }
  xv[i]=float(val); acc+=xv[i]*xv[i];
}
acc=simd_sum(acc);
if(sg==0)partial[lane]=0;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane==0)partial[sg]=acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) { acc=simd_sum(partial[lane]); if(lane==0)invs[row]=metal::precise::rsqrt(acc/float(H)+eps); }
