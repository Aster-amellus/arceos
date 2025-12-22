// modules/axsync/src/rcu.rs

extern crate alloc;

use alloc::boxed::Box;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};

use axconfig::plat::CPU_NUM;
use axhal::percpu::this_cpu_id;
use kernel_guard::NoPreempt;
use kspin::SpinNoIrq;

/// Epoch type used by EBR.
pub type Epoch = u64;

/// Global epoch (monotonic increasing).
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(1);

/// EBR uses multiple "generations" for safe reclamation.
/// A common engineering choice is 3 slots, reclaiming (e-2) generation conservatively.
pub const EBR_GENS: usize = 3;

/// A type-erased reclaim callback.
pub type ReclaimFn = unsafe fn(*mut u8);

/// A retired object record.
#[repr(C)]
pub struct Retired {
    pub ptr: *mut u8,
    pub reclaim: ReclaimFn,
    pub retire_epoch: Epoch,
}

/// Per-CPU retire queue backed by an IRQ-safe spin lock.
///
/// Bring-up uses a `Vec<Retired>` to keep pin/unpin fast paths allocation free.
pub struct RetireQueue {
    inner: SpinNoIrq<Vec<Retired>>,
}

impl RetireQueue {
    pub const fn new() -> Self {
        Self {
            inner: SpinNoIrq::new(Vec::new()),
        }
    }

    fn push(&self, retired: Retired) {
        self.inner.lock().push(retired);
    }

    /// Reclaim up to `budget` entries whose retire epoch is at least two
    /// generations older than the current global epoch.
    fn reclaim(&self, current_epoch: Epoch, budget: usize) -> usize {
        let mut guard = self.inner.lock();
        let mut reclaimed = 0;
        let mut idx = 0;
        while idx < guard.len() && reclaimed < budget {
            if guard[idx].retire_epoch + 2 <= current_epoch {
                let retired = guard.remove(idx);
                unsafe { (retired.reclaim)(retired.ptr) };
                reclaimed += 1;
            } else {
                idx += 1;
            }
        }
        reclaimed
    }

    fn len(&self) -> usize {
        self.inner.lock().len()
    }
}

/// Per-CPU EBR record storing nesting/active state and retire queues.
#[repr(C)]
pub struct EbrCpuRecord {
    pub nesting: AtomicU32,
    pub active: AtomicBool,
    pub announced: AtomicU64,
    pub retire: [RetireQueue; EBR_GENS],
}

#[percpu::def_percpu]
static EBR_CPU: EbrCpuRecord = EbrCpuRecord {
    nesting: AtomicU32::new(0),
    active: AtomicBool::new(false),
    announced: AtomicU64::new(0),
    retire: [RetireQueue::new(), RetireQueue::new(), RetireQueue::new()],
};

/// Published per-CPU participation state (scannable by other CPUs).
static CPU_ACTIVE: [AtomicBool; CPU_NUM] = [const { AtomicBool::new(false) }; CPU_NUM];
static CPU_ANNOUNCED: [AtomicU64; CPU_NUM] = [const { AtomicU64::new(0) }; CPU_NUM];
static CPU_NESTING: [AtomicU32; CPU_NUM] = [const { AtomicU32::new(0) }; CPU_NUM];

/// Serializes epoch advance for bring-up simplicity.
static ADVANCE_LOCK: SpinNoIrq<()> = SpinNoIrq::new(());

/// Guard that pins the current CPU (preemption disabled) until dropped.
pub struct Guard {
    _np: NoPreempt,
    cookie: GuardCookie,
}

#[derive(Clone, Copy)]
struct GuardCookie {
    cpu_id: usize,
    outermost: bool,
}

impl GuardCookie {
    fn new(cpu_id: usize, outermost: bool) -> Self {
        Self { cpu_id, outermost }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let cid = self.cookie.cpu_id;
        let prev = CPU_NESTING[cid].fetch_sub(1, Ordering::Relaxed);
        current_record().nesting.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(prev > 0);
        if self.cookie.outermost {
            CPU_ACTIVE[cid].store(false, Ordering::Release);
        }
    }
}

fn current_record() -> &'static EbrCpuRecord {
    unsafe { EBR_CPU.current_ref_raw() }
}

fn global_epoch() -> Epoch {
    GLOBAL_EPOCH.load(Ordering::Relaxed)
}

/// Enter the read-side critical section. The returned guard drops to unpin.
pub fn pin() -> Guard {
    let _np = NoPreempt::new();
    let cid = this_cpu_id();
    let prev = CPU_NESTING[cid].fetch_add(1, Ordering::Relaxed);
    current_record().nesting.fetch_add(1, Ordering::Relaxed);
    let outermost = prev == 0;
    if outermost {
        let epoch = global_epoch();
        CPU_ANNOUNCED[cid].store(epoch, Ordering::Relaxed);
        CPU_ACTIVE[cid].store(true, Ordering::Release);
        current_record().announced.store(epoch, Ordering::Relaxed);
        current_record().active.store(true, Ordering::Relaxed);
    }
    Guard {
        _np,
        cookie: GuardCookie::new(cid, outermost),
    }
}

/// Returns whether the current CPU is inside a pinned section.
pub fn is_pinned() -> bool {
    let cid = this_cpu_id();
    CPU_NESTING[cid].load(Ordering::Relaxed) > 0
}

/// Compatibility helpers for existing callers.
pub fn rcu_read_lock() -> Guard {
    pin()
}

pub fn rcu_read_unlock(_guard: Guard) {}

/// Publish a pointer with Release ordering so that initialization is visible to RCU readers.
pub fn rcu_assign_ptr<T>(dst: &AtomicPtr<T>, new: *mut T) {
    dst.store(new, Ordering::Release);
}

/// Dereference an RCU pointer with Acquire ordering.
/// Callers must hold a [`Guard`] while dereferencing the returned value.
pub fn rcu_deref_ptr<T>(src: &AtomicPtr<T>) -> *mut T {
    src.load(Ordering::Acquire)
}

fn retire_queue_for(epoch: Epoch) -> &'static RetireQueue {
    let idx = (epoch as usize) % EBR_GENS;
    &current_record().retire[idx]
}

/// Retire a raw pointer with an explicit reclaim callback.
///
/// # Safety
///
/// The pointer must be valid for the provided callback and must not be reused elsewhere.
pub unsafe fn retire_ptr(ptr: *mut u8, reclaim: ReclaimFn) {
    let epoch = global_epoch();
    retire_queue_for(epoch).push(Retired {
        ptr,
        reclaim,
        retire_epoch: epoch,
    });
}

/// Retire a typed pointer using `Box::from_raw` to reclaim.
pub unsafe fn retire<T>(ptr: *mut T) {
    unsafe fn reclaim<T>(ptr: *mut u8) {
        let _ = Box::from_raw(ptr as *mut T);
    }

    retire_ptr(ptr as *mut u8, reclaim::<T>);
}

fn try_advance_epoch() -> bool {
    let _lock = ADVANCE_LOCK.lock();
    let cur = global_epoch();
    for cpu_id in 0..CPU_NUM {
        let active = CPU_ACTIVE[cpu_id].load(Ordering::Acquire);
        if active {
            let announced = CPU_ANNOUNCED[cpu_id].load(Ordering::Relaxed);
            if announced != cur {
                return false;
            }
        }
    }
    GLOBAL_EPOCH.store(cur + 1, Ordering::Relaxed);
    true
}

fn reclaim_oldest_generation(budget: usize) -> usize {
    let cur = global_epoch();
    if cur < 2 {
        return 0;
    }
    let target_epoch = cur - 2;
    let queue = retire_queue_for(target_epoch);
    queue.reclaim(cur, budget)
}

fn poll_with_budget(budget: usize) -> bool {
    // Step 1: reclaim from the oldest generation visible to this CPU.
    let reclaimed = reclaim_oldest_generation(budget);

    // Step 2: attempt to advance the global epoch.
    let advanced = try_advance_epoch();

    // Step 3: if we advanced, reclaim again (respecting the budget window).
    let reclaimed_after_advance = if advanced {
        reclaim_oldest_generation(budget.saturating_sub(reclaimed))
    } else {
        0
    };

    reclaimed > 0 || reclaimed_after_advance > 0 || advanced
}

/// Non-blocking poll that amortizes reclamation work.
pub fn poll() -> bool {
    poll_with_budget(usize::MAX)
}

/// A budgeted variant suited for timer tick hooks.
pub fn poll_budgeted(budget: usize) -> bool {
    if budget == 0 {
        return false;
    }
    poll_with_budget(budget)
}

/// Block until a grace period has elapsed.
///
/// # Panics
///
/// Panics if invoked while pinned.
pub fn synchronize() {
    assert!(!is_pinned(), "synchronize() called inside pinned section");
    let start = global_epoch();
    let target = start + 2;
    while global_epoch() < target {
        poll();
        spin_loop();
    }
}

/// Debug snapshot of per-CPU state.
#[cfg(feature = "rcu-debug")]
#[derive(Debug, Default, Clone)]
pub struct Snapshot {
    pub global_epoch: Epoch,
    pub per_cpu: [CpuSnapshot; CPU_NUM],
}

#[cfg(feature = "rcu-debug")]
#[derive(Debug, Default, Clone)]
pub struct CpuSnapshot {
    pub active: bool,
    pub nesting: u32,
    pub announced: Epoch,
    pub retire_lens: [usize; EBR_GENS],
}

#[cfg(feature = "rcu-debug")]
pub fn snapshot() -> Snapshot {
    let global_epoch = global_epoch();
    let mut per_cpu = [CpuSnapshot::default(); CPU_NUM];
    for cpu_id in 0..CPU_NUM {
        let record = unsafe { &*EBR_CPU.remote_ref_raw(cpu_id) };
        per_cpu[cpu_id] = CpuSnapshot {
            active: CPU_ACTIVE[cpu_id].load(Ordering::Relaxed),
            nesting: CPU_NESTING[cpu_id].load(Ordering::Relaxed),
            announced: CPU_ANNOUNCED[cpu_id].load(Ordering::Relaxed),
            retire_lens: [
                record.retire[0].len(),
                record.retire[1].len(),
                record.retire[2].len(),
            ],
        };
    }
    Snapshot {
        global_epoch,
        per_cpu,
    }
}

#[cfg(feature = "rcu-debug")]
pub fn check_invariants() -> &'static str {
    for cpu_id in 0..CPU_NUM {
        if CPU_NESTING[cpu_id].load(Ordering::Relaxed) == 0
            && CPU_ACTIVE[cpu_id].load(Ordering::Relaxed)
        {
            return "nesting=0 but active=true";
        }
    }
    "ok"
}
