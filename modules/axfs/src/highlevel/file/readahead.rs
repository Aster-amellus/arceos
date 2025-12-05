use alloc::sync::Arc;

use axfs_ng_vfs::{FileNode, FileNodeOps, VfsResult};
use lru::LruCache;

use super::PAGE_SIZE;
use crate::{CachedFile, PageCache, highlevel::file::CachedFileShared};

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

fn do_sync_prefetch(
    shared: &CachedFileShared,
    in_memory: bool,
    file: &dyn FileNodeOps,
    ra_state: ReadaheadState,
) -> VfsResult<()> {
    // FIXME: we won't fetch pn that user request either
    // If size is 0, we do nothing here. The caller (prefetch_page) must handle the demand read.
    if ra_state.size == 0 {
        return Ok(());
    }
    let pg_readahead_trigger = ra_state.get_trigger_offset();

    for pn in ra_state.start_pn..(ra_state.start_pn + ra_state.size) {
        // 1. Check if page exists (Optimistic check)
        {
            let caches = shared.page_cache.lock();
            if caches.contains(&pn) {
                continue;
            }
        } // Lock released here

        // 2. Prepare page and perform I/O without holding the lock
        // Note: We might do redundant work if another thread loads it meanwhile,
        // but it's better than blocking the whole cache during I/O.
        let mut page = PageCache::new()?;

        // Set the flag on the trigger page
        if pn == pg_readahead_trigger {
            page.pg_readahead = true;
        }

        if in_memory {
            page.data().fill(0);
        } else {
            file.read_at(page.data(), pn as u64 * PAGE_SIZE as u64)?;
        }

        // 3. Insert into cache
        let mut caches = shared.page_cache.lock();
        // Check again to avoid overwriting if someone else loaded it
        if !caches.contains(&pn) {
            if caches.len() == caches.cap().get() {
                if let Some((evict_pn, mut evicted_page)) = caches.pop_lru() {
                    if evicted_page.dirty {
                        // We must drop the lock before writing back to avoid deadlock/blocking
                        drop(caches);
                        file.write_at(evicted_page.data(), evict_pn as u64 * PAGE_SIZE as u64)?;
                        // Re-acquire lock to continue
                        caches = shared.page_cache.lock();
                    }
                }
            }
            // Re-check capacity after potential re-acquire
            if caches.len() < caches.cap().get() {
                caches.put(pn, page);
            }
        }
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

pub(super) trait Readahead {
    // fn prefetch_page<'a>(
    //     &'a self,
    //     file: &FileNode,
    //     start: u32,
    //     read_size: u32,
    //     pn: u32,
    // ) -> VfsResult<()>;

    /// find cache from cache
    fn find_page_from_cache<'a>(
        &self,
        caches: &'a mut LruCache<u32, PageCache>,
        pn: u32,
    ) -> Option<(&'a mut PageCache, bool)>;
}

impl Readahead for CachedFile {
    fn find_page_from_cache<'a>(
        &self,
        caches: &'a mut LruCache<u32, PageCache>,
        pn: u32,
    ) -> Option<(&'a mut PageCache, bool)> {
        match caches.get_mut(&pn) {
            Some(cache) => {
                let mut should_async_readahead = false;
                if cache.pg_readahead {
                    cache.pg_readahead = false;
                    should_async_readahead = true;
                    self.ra_state.lock().subsequent_readahead();
                }
                Some((cache, should_async_readahead))
            }
            None => None,
        }
    }

    // fn prefetch_page<'a>(
    //     &'a self,
    //     file: &FileNode,
    //     start: u32,
    //     read_size: u32,
    //     pn: u32,
    // ) -> VfsResult<()> {
    //     let mut need_prefetch = false;
    //     let mut need_async = false;

    //     // 1. Check cache hit and readahead flag
    //     if let Some(cache) = self.shared.page_cache.lock().get_mut(&pn) {
    //         if cache.pg_readahead {
    //             cache.pg_readahead = false;
    //             self.ra_state.lock().subsequent_readahead();
    //             need_prefetch = true;
    //             need_async = true;
    //         }
    //     } else {
    //         // 2. Cache miss: Analyze pattern and setup initial window
    //         use AccessPattern::*;
    //         let mut ra_state = self.ra_state.lock(); // Fixed: removed &mut
    //         match AccessPattern::check(&ra_state, start, read_size) {
    //             Sequential | Beginning | Unaligned => {
    //                 // Linux treats unaligned sequential reads (overlapping previous read)
    //                 // as sequential. On a miss, we restart the window.
    //                 ra_state.initial_readahead(pn, read_size);
    //             }
    //             LargeRead | Random => {
    //                 // For random or very large reads, disable readahead to avoid cache pollution
    //                 ra_state.size = 0;
    //             }
    //         }
    //         need_prefetch = true;
    //     }

    //     // 3. Perform prefetch if needed (borrow of caches is released above)
    //     if need_prefetch {
    //         let ra_state = *self.ra_state.lock();
    //         let in_memory = self.in_memory;
    //         let file_ops = file.inner().clone(); // Get Arc<dyn FileNodeOps>
    //         if need_async {
    //             let shared = self.shared.clone();
    //             axtask::spawn(move || {
    //                 do_sync_prefetch(&shared, in_memory, file_ops.as_ref(), ra_state);
    //             });
    //         } else {
    //             do_sync_prefetch(&self.shared, in_memory, file_ops.as_ref(), ra_state)?;
    //         }
    //     }

    //     Ok(())
    // }
}

pub fn async_prefetch(
    cache_shared: Arc<CachedFileShared>,
    file: Arc<dyn FileNodeOps>,
    start_pn: u32,
    size: u32,
) {
    unimplemented!()
}
