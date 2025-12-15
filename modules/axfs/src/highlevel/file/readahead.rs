use alloc::{sync::Arc, vec::Vec};

use axfs_ng_vfs::{FileNode, FileNodeOps, VfsResult};
use lru::LruCache;

use super::*;

/// default max page size (128KB / 4KB = 32 pages)
pub const RA_MAX_PAGES: u32 = 32;

/// default min page size (prevent too small IO)
pub const RA_MIN_PAGES: u32 = 2; // Linux VM_MIN_READAHEAD

/// ramp up initial scale factor
const INIT_RA_SCALE: u32 = 4;

/// For small reads (e.g. 4K/16K), avoid scheduling overhead by doing sync-only
/// readahead (no PG_readahead / async prefetch trigger).
const SYNC_ONLY_MAX_REQ_PAGES: u32 = 4;

/// Minimum sync readahead window for small reads.
///
/// This reduces virtio/QEMU small-IO overhead for 4K sequential workloads.
const SMALL_REQ_MIN_RA_PAGES: u32 = 16;

/// ramp up factor for subsequent readahead
const RAMP_UP_SCALE: u32 = 2;

/// Threshold in pages (16KB = 4 * 4KB pages). For requests no larger than this,
/// if the shared bounce buffer is contended, fall back to a private allocation
/// to minimize tail latency.
const PRIVATE_ALLOC_THRESHOLD: u32 = 4;

/// Upper bound for private bounce buffer allocation under contention.
///
/// This is set to the default readahead window size, so the worst-case private
/// allocation is bounded (e.g. 32 pages = 128KB).
const PRIVATE_ALLOC_ON_CONTEND_MAX_PAGES: usize = RA_MAX_PAGES as usize;

pub struct ReadaheadState {
    pub start_pn: u32,
    pub size: u32,
    pub async_size: u32,
    pub prev_pn: u32,
    pub max_pages: u32,
}

impl ReadaheadState {
    /// create an empty readahead state
    pub const fn new(max_pages: u32) -> Self {
        Self {
            start_pn: 0,
            size: 0, // disabled readahead for now
            async_size: 0,
            prev_pn: 0,
            max_pages,
        }
    }

    /// Updates readahead window upon a cache miss, and if it should trigger sync
    /// readahead, returns the sync readahead parameters.

    /// # Returns
    /// - `Some((start_pn, size, pg_readahead_offset))` if readahead should be
    /// triggered, and sync readahead should be submitted.
    /// - `None` if random access is detected, and no readahead should be triggered,
    /// *but* the user requested page should still be submitted for IO.
    pub fn update_window_on_cache_miss(
        &mut self,
        trigger_pn: u32,
        req_size: u32,
    ) -> Option<(u32, u32, u32)> {
        // request size exceeds max page limit
        if req_size > self.max_pages {
            self.start_pn = trigger_pn;
            self.size = self.max_pages;
            self.async_size = self.max_pages - 1; // set the next page as PG_readahead to launch async readahead immediately
            Some((trigger_pn, self.max_pages, self.get_trigger_offset()))
        } else if trigger_pn == self.prev_pn + 1 || trigger_pn == self.prev_pn // sequential access, even though the PG_readahead flag was missed
            || trigger_pn == 0
        // first access
        {
            self.start_pn = trigger_pn;
            let new_size = Self::init_ra_size(req_size, self.max_pages);
            // NOTE: why ramping up aggressively may cause lower speed?
            // if self.size > 0 {
            //     new_size = new_size.saturating_mul(RAMP_UP_SCALE).min(self.max_pages);
            // }
            self.size = new_size;
            if req_size <= SYNC_ONLY_MAX_REQ_PAGES {
                // Sync-only: don't set PG_readahead so we won't spawn/yield per 4K op.
                self.async_size = 0;
            } else {
                self.async_size = new_size - req_size;
            }
            Some((trigger_pn, new_size, self.get_trigger_offset()))
        }
        // else if trigger_pn >= self.start_pn && trigger_pn < self.start_pn + self.size {
        //     // TODO: add thrashing detection here
        // self.start_pn = trigger_pn;
        // }
        else {
            self.start_pn = trigger_pn;
            // random access, reset readahead window
            self.size = 0;
            self.async_size = 0;
            None
        }
    }

    /// hit PG_readahead, prepare window for upcoming async prefetch,
    /// async prefetch will be triggered at once after this function call.
    pub fn update_window_for_async(&mut self) {
        //  new window starts from previous one's next page
        self.start_pn = self.start_pn.saturating_add(self.size);
        let mut new_size = self.size.saturating_mul(RAMP_UP_SCALE);
        new_size = new_size.min(self.max_pages);
        self.size = new_size;
        // keep full pipline
        self.async_size = new_size;
    }

    /// get PG_readahead offset for async prefetch
    /// formula: start + size - async_size
    pub const fn get_trigger_offset(&self) -> u32 {
        if self.size == 0 || self.async_size == 0 {
            return u32::MAX;
        }
        self.start_pn + self.size - self.async_size
    }

    /// update history after the whole read request is processed
    pub const fn update_history(&mut self, last_read_pn: u32) {
        self.prev_pn = last_read_pn;
    }

    /// create initial readahead size based on request size
    const fn init_ra_size(req_size: u32, max_pages: u32) -> u32 {
        let mut size = req_size.saturating_mul(INIT_RA_SCALE);
        if req_size <= SYNC_ONLY_MAX_REQ_PAGES && size < SMALL_REQ_MIN_RA_PAGES {
            size = SMALL_REQ_MIN_RA_PAGES;
        }
        if size < RA_MIN_PAGES {
            size = RA_MIN_PAGES;
        } else if size > max_pages {
            size = max_pages;
        }
        size
    }
}

pub(super) trait Readahead {
    /// find page from cache
    /// # Returns
    /// - `Some` if cache hit, along with [PageCache] mut reference and `Some((start_pn, size, pg_readahead_offset))` if PG_readahead flag is found
    /// - `None` if cache miss
    fn find_page_from_cache<'a>(
        &self,
        caches: &'a mut LruCache<u32, PageCache>,
        pn: u32,
    ) -> Option<(&'a mut PageCache, Option<(u32, u32, u32)>)>;
}

impl Readahead for CachedFile {
    fn find_page_from_cache<'a>(
        &self,
        caches: &'a mut LruCache<u32, PageCache>,
        pn: u32,
    ) -> Option<(&'a mut PageCache, Option<(u32, u32, u32)>)> {
        caches.get_mut(&pn).map(|cache| {
            // cache hit
            let mut new_pg_pn = None;
            if cache.pg_readahead {
                // find PG_readahead flag, clear the flag and prepare for async prefetch
                cache.pg_readahead = false;
                let mut ra = self.ra_state.lock();
                if ra.size > 0 {
                    ra.update_window_for_async();
                    new_pg_pn = Some((ra.start_pn, ra.size, ra.get_trigger_offset()));
                }
            }
            (cache, new_pg_pn)
        })
    }
}

pub fn async_prefetch(
    cache_shared: Arc<CachedFileShared>,
    file: Arc<dyn FileNodeOps>,
    in_memory: bool,
    start_pn: u32,
    size: u32,
    async_pg_pn: u32,
) -> VfsResult<()> {
    let file = FileNode::new(file);
    io_submit(&cache_shared, &file, in_memory, start_pn, size, async_pg_pn)
}

pub fn io_submit(
    cache_shared: &CachedFileShared,
    file: &FileNode,
    in_memory: bool,
    start_pn: u32,
    size: u32,
    async_pg_pn: u32,
) -> VfsResult<()> {
    // 1) Insert pending placeholders into the page cache.
    // Only pages inserted into `owned` will be filled by this call.
    let mut owned: Vec<(u32, Arc<PendingPage>)> = Vec::new();
    {
        let mut caches = cache_shared.page_cache.lock();
        for pn in start_pn..(start_pn + size) {
            if let Some(existing) = caches.get_mut(&pn) {
                if existing.pending().is_some() {
                    cache_shared
                        .inflight_hit
                        .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                }
                continue;
            }

            if caches.len() == caches.cap().get() {
                if let Some((evict_pn, mut evicted_page)) = caches.pop_lru() {
                    // Avoid evicting in-flight pages.
                    if evicted_page.pending().is_some() {
                        caches.put(evict_pn, evicted_page);
                        return Ok(());
                    }
                    let _ = cache_shared.evict_cache(file, evict_pn, &mut evicted_page);
                }
            }

            let pending = cache_shared.alloc_pending();
            let mut page = PageCache::new()?;
            page.set_pending(pending.clone());
            caches.put(pn, page);
            cache_shared
                .inflight_new
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            owned.push((pn, pending));
        }
    }

    if owned.is_empty() {
        return Ok(());
    }

    // In-memory file: populate cache with zeros, no device IO.
    if in_memory {
        let mut caches = cache_shared.page_cache.lock();
        for (pn, pending) in owned {
            if let Some(page) = caches.get_mut(&pn) {
                if pn == async_pg_pn {
                    page.pg_readahead = true;
                }
                page.data().fill(0);
                page.clear_pending();
            }
            pending.complete_ok();
            cache_shared.recycle_pending(pending);
        }
        return Ok(());
    }

    let first_pn = owned.first().unwrap().0;
    let last_pn = owned.last().unwrap().0;
    let span_pages = (last_pn - first_pn + 1) as usize;

    // 2) IO worker: does lock-free device read and then fills the page cache.
    let io_worker = |bounce_buffer: &mut [u8]| -> VfsResult<()> {
        file.read_at(bounce_buffer, first_pn as u64 * PAGE_SIZE as u64)?;

        let mut caches = cache_shared.page_cache.lock();
        for &(pn, _) in &owned {
            let Some(page) = caches.get_mut(&pn) else {
                continue;
            };

            if pn == async_pg_pn {
                page.pg_readahead = true;
            }
            let offset = (pn - first_pn) as usize * PAGE_SIZE;
            page.data()
                .copy_from_slice(&bounce_buffer[offset..offset + PAGE_SIZE]);
            page.clear_pending();
        }
        Ok(())
    };

    // 3) Hybrid strategy: prefer shared buffer; under contention, allocate a bounded
    // private buffer to avoid serializing foreground reads with background prefetch.
    unsafe {
        let res = if let Some(mut guard) = cache_shared.bounce_buffer.try_lock() {
            // Fast path: reuse shared buffer (no allocation).
            let bounce_buffer = guard
                .as_mut_slice()
                .get_unchecked_mut(0..span_pages * PAGE_SIZE);
            io_worker(bounce_buffer)
        } else if span_pages <= PRIVATE_ALLOC_THRESHOLD as usize {
            // Small request: allocate private buffer to avoid queuing/yield overhead.
            let mut bounce_buffer = Vec::with_capacity(span_pages * PAGE_SIZE);
            bounce_buffer.set_len(span_pages * PAGE_SIZE);
            io_worker(bounce_buffer.as_mut_slice())
        } else if span_pages <= PRIVATE_ALLOC_ON_CONTEND_MAX_PAGES {
            // Contended: keep memory bounded but avoid blocking behind other IO.
            let mut bounce_buffer = Vec::with_capacity(span_pages * PAGE_SIZE);
            bounce_buffer.set_len(span_pages * PAGE_SIZE);
            io_worker(bounce_buffer.as_mut_slice())
        } else {
            // Large request: wait for the shared buffer to avoid large allocations.
            let mut guard = cache_shared.bounce_buffer.lock();
            let bounce_buffer = guard
                .as_mut_slice()
                .get_unchecked_mut(0..span_pages * PAGE_SIZE);
            io_worker(bounce_buffer)
        };

        // Complete in-flight entries.
        match res {
            Ok(()) => {
                for (_pn, pending) in owned {
                    pending.complete_ok();
                    cache_shared.recycle_pending(pending);
                }
                Ok(())
            }
            Err(err) => {
                // Remove placeholders so future reads can retry.
                let mut caches = cache_shared.page_cache.lock();
                for (pn, pending) in owned {
                    let _ = caches.pop(&pn);
                    pending.complete_err(VfsError::InvalidInput);
                    cache_shared.recycle_pending(pending);
                }
                Err(err)
            }
        }
    }
}
