use axfs_ng_vfs::{FileNode, VfsResult};
use lru::LruCache;

use super::PAGE_SIZE;
use crate::PageCache;

const READAHEAD_MAX_SIZE: u32 = 32; // max_readahead pages (e.g., 128KB / 4KB)
const INIT_SCALE: u32 = 2; // scale_0 (2 or 4) 
const RAMP_UP_SCALE: u32 = 2; // subsequent scale (2) 

#[derive(Debug, Clone, Copy)]
pub struct ReadaheadState {
    /// The start page index of the current readahead window
    pub start_pn: u32,
    /// Number of pages in the current window
    pub size: u32,
    /// Threshold to trigger async readahead
    /// (Trigger when remaining pages < async_size)
    pub async_size: u32,
    /// The last accessed page index for sequential check
    pub prev_pn: u32,
}

impl ReadaheadState {
    /// Create a disabled/empty state
    pub const fn new() -> Self {
        Self {
            start_pn: 0,
            size: 0, // 0 means readahead is disabled/inactive
            async_size: 0,
            prev_pn: 0,
        }
    }

    /// Update the access history, should be called at the end of every read
    pub fn update_history(&mut self, current_read_end: u32) {
        self.prev_pn = current_read_end;
    }

    /// Triggered by Cache miss
    pub fn initial_readahead(&mut self, pn_offset: u32, read_size: u32) {
        self.size = (read_size * INIT_SCALE).min(READAHEAD_MAX_SIZE);
        self.async_size = self.size - read_size;

        // Window starts exactly where we missed
        self.start_pn = pn_offset;
    }

    /// Triggered by hitting a page with PG_readahead flag
    pub fn subsequent_readahead(&mut self) {
        // The new window starts immediately after the current one
        self.start_pn = self.start_pn + self.size;
        self.size = (self.size * RAMP_UP_SCALE).min(READAHEAD_MAX_SIZE);

        // full pipelining
        self.async_size = self.size;
    }

    /// Get the offset page number to trigger async readahead
    pub const fn get_trigger_offset(&self) -> u32 {
        if self.size == 0 {
            return 0;
        }
        self.start_pn + self.size - self.async_size
    }
}

/// - start: the start pn from user read request,
/// - pn the currrent pn to read to dst
/// - read_size: the size of user read request
pub fn prefetch_page<'a>(
    file: &FileNode,
    ra_state: &'a mut ReadaheadState,
    in_memory: bool,
    caches: &'a mut LruCache<u32, PageCache>,
    start: u32,
    read_size: u32,
    pn: u32,
) -> VfsResult<&'a mut PageCache> {
    let mut need_prefetch = false;
    // 1. Check cache hit and readahead flag
    if let Some(cache) = caches.get_mut(&pn) {
        if cache.pg_readahead {
            cache.pg_readahead = false;
            ra_state.subsequent_readahead();
            need_prefetch = true;
        }
    } else {
        // 2. Cache miss: Analyze pattern and setup initial window
        use AccessPattern::*;
        match AccessPattern::check(ra_state, start, read_size) {
            Sequential | Beginning | Unaligned => {
                // Linux treats unaligned sequential reads (overlapping previous read)
                // as sequential. On a miss, we restart the window.
                ra_state.initial_readahead(pn, read_size);
            }
            LargeRead | Random => {
                // For random or very large reads, disable readahead to avoid cache pollution
                ra_state.size = 0;
            }
        }
        need_prefetch = true;
    }

    // 3. Perform prefetch if needed (borrow of caches is released above)
    if need_prefetch {
        do_sync_prefetch(caches, in_memory, file, ra_state)?;
    }

    // 4. Ensure the requested page is loaded (if prefetch didn't load it)
    if !caches.contains(&pn) {
        if caches.len() == caches.cap().get() {
            if let Some((evict_pn, mut page)) = caches.pop_lru() {
                if page.dirty {
                    file.write_at(page.data(), evict_pn as u64 * PAGE_SIZE as u64)?;
                }
            }
        }
        let mut page = PageCache::new()?;
        if in_memory {
            page.data().fill(0);
        } else {
            file.read_at(page.data(), pn as u64 * PAGE_SIZE as u64)?;
        }
        caches.put(pn, page);
    }

    Ok(caches.get_mut(&pn).unwrap())
}

fn do_sync_prefetch(
    caches: &mut LruCache<u32, PageCache>,
    in_memory: bool,
    file: &FileNode,
    ra_state: &ReadaheadState,
) -> VfsResult<()> {
    if ra_state.size == 0 {
        return Ok(());
    }
    let pg_readahead_trigger = ra_state.get_trigger_offset();
    for pn in ra_state.start_pn..(ra_state.start_pn + ra_state.size) {
        // Use get() to update LRU position if page exists
        if caches.get(&pn).is_some() {
            continue;
        }
        
        if caches.len() == caches.cap().get() {
            if let Some((pn, mut page)) = caches.pop_lru() {
                if page.dirty {
                    file.write_at(page.data(), pn as u64 * PAGE_SIZE as u64)?;
                }
            }
        }
        let mut page = PageCache::new()?;
        if in_memory {
            page.data().fill(0);
        } else {
            file.read_at(page.data(), pn as u64 * PAGE_SIZE as u64)?;
        }
        caches.put(pn, page);
    }
    
    // Set the flag on the trigger page
    if let Some(cache) = caches.get_mut(&pg_readahead_trigger) {
        cache.pg_readahead = true;
    }
    Ok(())
}

enum AccessPattern {
    Sequential,
    Beginning,
    Unaligned,
    LargeRead,
    Random,
}

impl AccessPattern {
    fn check(ra_state: &ReadaheadState, start: u32, read_size: u32) -> Self {
        let prev_page = ra_state.prev_pn;
        use AccessPattern::*;
        if start == 0 {
            Beginning
        } else if start == prev_page + 1 {
            Sequential
        } else if start == prev_page {
            Unaligned
        } else if read_size > READAHEAD_MAX_SIZE {
            LargeRead
        } else {
            Random
        }
    }
}
