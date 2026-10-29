// The rows' recurrence inputs K1S holds in threadgroup memory, in K1R's layout (f32): k [TL, HK, DK] once per k head,
// v [TL, HV, DV], g | beta [TL, HV, 2].
{
  const uint tid = sg * 32u + lane;
  if (hv % GRP == 0) for (uint e = tid; e < uint(TL * DK); e += uint(NT)) pk[((e / uint(DK)) * HK + hk) * DK + e % uint(DK)] = ks[e / uint(DK)][e % uint(DK)];
  for (uint e = tid; e < uint(TL * DV); e += uint(NT)) pv[((e / uint(DV)) * HV + hv) * DV + e % uint(DV)] = vs[e / uint(DV)][e % uint(DV)];
  for (uint e = tid; e < uint(2 * TL); e += uint(NT)) pg[((e / 2u) * HV + hv) * 2u + e % 2u] = gb[e / 2u][e % 2u];
}