//! A minimal QSBR (Quiescent State Based Reclamation) RCU implementation.
//!
//! Design notes (important constraints):
//! - Read-side critical sections must not block or context-switch.
//! - We implement this by disabling both local IRQs and preemption with
//!   [`kernel_guard::NoPreemptIrqSave`].
//! - Quiescent states are reported by the scheduler/timer-tick hook in `axtask`.

use core::sync::atomic::{AtomicU64, Ordering};

use axhal::percpu::this_cpu_id;
use kernel_guard::NoPreemptIrqSave;

use alloc::{boxed::Box, vec::Vec};
use kspin::SpinNoIrq;

/// Global epoch counter.
///
/// Writer bumps it to start a new grace period, and waits until all CPUs
/// report a quiescent state observed at or after the target epoch.
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "rcu-debug")]
static QS_REPORTS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "rcu-debug")]
static SYNC_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "rcu-debug")]
static SYNC_SPINS: AtomicU64 = AtomicU64::new(0);

const MAX_CPUS: usize = axconfig::plat::CPU_NUM;

/// Per-CPU "last observed epoch at quiescent state".
///
/// NOTE: This is a static array to avoid coupling with `axhal`'s percpu storage.
static CPU_QS_EPOCH: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];

#[inline(always)]
fn this_cpu_state() -> &'static AtomicU64 {
    &CPU_QS_EPOCH[this_cpu_id()]
}

/// Read-side guard.
///
/// Holding this guard guarantees:
/// - local IRQs are disabled
/// - preemption is disabled
///
/// Therefore the current task cannot be context-switched, which satisfies the
/// QSBR assumption for readers.
pub struct RcuReadGuard {
    _guard: NoPreemptIrqSave,
}

/// Enter an RCU read-side critical section.
#[inline(always)]
pub fn rcu_read_lock() -> RcuReadGuard {
    let guard = NoPreemptIrqSave::new();
    core::sync::atomic::compiler_fence(Ordering::SeqCst);
    RcuReadGuard { _guard: guard }
}

impl Drop for RcuReadGuard {
    #[inline(always)]
    fn drop(&mut self) {
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        // `_guard` drops here and restores irq/preempt state.
    }
}

/// Report a quiescent state for the current CPU.
///
/// This is expected to be called by the scheduler/timer hook.
#[inline(always)]
pub fn rcu_report_qs() {
    let global = GLOBAL_EPOCH.load(Ordering::Acquire);
    this_cpu_state().store(global, Ordering::Release);

    #[cfg(feature = "rcu-debug")]
    QS_REPORTS.fetch_add(1, Ordering::Relaxed);
}

/// Wait for a grace period.
///
/// # Safety/Constraints
/// - Must NOT be called while holding [`RcuReadGuard`].
/// - Must be called from a context where scheduling is allowed.
pub fn synchronize_rcu() {
    #[cfg(feature = "rcu-debug")]
    SYNC_CALLS.fetch_add(1, Ordering::Relaxed);

    // Move to the next epoch.
    let target_epoch = GLOBAL_EPOCH.fetch_add(1, Ordering::SeqCst) + 1;

    // The current CPU is not inside a read-side critical section (otherwise we'd deadlock).
    // Proactively report QS to avoid waiting for ourselves.
    rcu_report_qs();

    // Only wait for CPUs that are actually online/active for scheduling.
    // Otherwise we'd wait forever for CPUs that are configured but not brought up.
    let active_cpus = axtask::active_cpu_num().min(MAX_CPUS);
    for cpu_id in 0..active_cpus {
        // Skip self: already reported above.
        if cpu_id == this_cpu_id() {
            continue;
        }

        let cpu_state = &CPU_QS_EPOCH[cpu_id];
        while cpu_state.load(Ordering::Acquire) < target_epoch {
            #[cfg(feature = "rcu-debug")]
            SYNC_SPINS.fetch_add(1, Ordering::Relaxed);

            // Cooperative backoff: let others run and reach a quiescent state.
            axtask::yield_now();
            core::hint::spin_loop();
        }
    }

    core::sync::atomic::fence(Ordering::SeqCst);
}

/// Debug counters snapshot.
#[cfg(feature = "rcu-debug")]
#[derive(Clone, Copy, Debug)]
pub struct RcuDebugSnapshot {
    pub global_epoch: u64,
    pub qs_reports: u64,
    pub sync_calls: u64,
    pub sync_spins: u64,
    pub active_cpus: usize,
    pub this_cpu: usize,
}

/// Returns current debug counters.
#[cfg(feature = "rcu-debug")]
pub fn debug_snapshot() -> RcuDebugSnapshot {
    RcuDebugSnapshot {
        global_epoch: GLOBAL_EPOCH.load(Ordering::Relaxed),
        qs_reports: QS_REPORTS.load(Ordering::Relaxed),
        sync_calls: SYNC_CALLS.load(Ordering::Relaxed),
        sync_spins: SYNC_SPINS.load(Ordering::Relaxed),
        active_cpus: axtask::active_cpu_num(),
        this_cpu: this_cpu_id(),
    }
}

/// A tiny micro-benchmark helper: measures ticks spent on `rcu_read_lock()`+drop.
///
/// Use only for rough comparisons (it includes loop overhead).
#[cfg(feature = "rcu-debug")]
pub fn bench_read_lock(iters: usize) -> u64 {
    let start = axhal::time::current_ticks();
    for _ in 0..iters {
        let _g = rcu_read_lock();
        core::hint::black_box(&0u8);
    }
    axhal::time::current_ticks().wrapping_sub(start)
}

type Callback = Box<dyn FnOnce() + Send + 'static>;

static DEFERRED_FREE_LIST: SpinNoIrq<Vec<Callback>> = SpinNoIrq::new(Vec::new());

/// Register a deferred drop callback (synchronous RCU-style reclamation).
#[inline]
pub fn call_rcu<T: Send + 'static>(data: T, drop_func: fn(T)) {
    let cb: Callback = Box::new(move || drop_func(data));
    DEFERRED_FREE_LIST.lock().push(cb);
}

/// Wait for a grace period and run all pending deferred callbacks.
pub fn rcu_barrier() {
    synchronize_rcu();

    let callbacks = {
        let mut guard = DEFERRED_FREE_LIST.lock();
        core::mem::take(&mut *guard)
    };

    for cb in callbacks {
        cb();
    }
}

// Auto-register the QS hook once constructors are run.
// This avoids axtask<->axsync cyclic dependency.
#[ctor_bare::register_ctor]
fn __axsync_rcu_register_hook() {
    axtask::register_quiescent_state_hook(rcu_report_qs);
}
