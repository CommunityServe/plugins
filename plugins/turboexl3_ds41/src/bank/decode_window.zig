//! Decode's window 0 at `window_rows` rows, not the widest decode route's 48, its memory the bank's decode slot rows
//! (read by the streamer's bill and grow).
//!
//! At the phase change the stream frees the prompt's transient scratch and allocates decode's window 0: the rows a
//! decode plan's transient loads take, sized for the widest decode route (48 ids, one window a geometry: 48 x (13.3 + 8.9
//! MB) = 1.066 GB on the 2.81 bpw pack, billed before the decode slot rows). A transient load is a miss the decode
//! admission declines (it ranks below every resident it could evict); the 16K card's greedy decode declines none, so the
//! window holds nothing. Window 0 holds `window_rows` rows, and a decode plan admits its declined misses past them over
//! the lowest-ranked residents the route does not use (`LayerPolicy.admitPast`); the bill gives the freed bytes to the
//! decode slot rows (about two more a layer). Taken only where every layer's decode rows are 2 * 48 or more (a resident
//! is always left to take), else the window stays the widest route's. Exact: a route's bytes are the records' wherever
//! they sit; only which rows hold them moves, and only on a route past the window.

/// Window 0's rows: room for a few declined misses a plan before any is admitted.
pub const window_rows: u32 = 4;
