// The finalize spliced after this (kernels/spec/mtp_top32_final.metal) writes its 32 ids into token_ids, which this
// launch keeps in threadgroup memory for the re-score and publishes as cands.
threadgroup uint token_ids[32];
