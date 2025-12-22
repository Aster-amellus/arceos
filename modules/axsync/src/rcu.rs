// modules/axsync/src/rcu.rs
#![allow(dead_code)]

use core::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering,
};

use kernel_guard::{NoPreempt, NoPreemptIrqSave};

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
/// (You can later switch to an intrusive node to avoid allocation.)
#[repr(C)]
pub struct Retired {
    pub ptr: *mut u8,
    pub reclaim: ReclaimFn,
    pub retire_epoch: Epoch,
}

/// Per-CPU EBR record.
///
/// Design notes:
/// - `nesting` must handle IRQ nesting (so use atomic RMW, not &mut-only updates).
/// - `active/announced` are read by other CPUs during epoch advance scans.
#[repr(C)]
pub struct EbrCpuRecord {
    /// Pin nesting count on this CPU.
    pub nesting: AtomicU32,
    /// Whether this CPU is currently in a pinned (read-side) critical section.
    pub active: AtomicBool,
    /// The epoch announced when entering the outermost critical section.
    pub announced: AtomicU64,

    /// Per-generation retire queues.
    ///
    /// You will choose the concrete container + lock:
    /// - intrusive list + irq-safe lock
    /// - Vec of Retired + lock (debug/bringup)
    pub retire: [RetireQueue; EBR_GENS],
}

/// Retire queue placeholder.
///
/// Replace internals with your chosen structure:
/// - SpinLock<Vec<Retired>> (bringup)
/// - Intrusive list (production)
pub struct RetireQueue {
    _private: (),
}

impl RetireQueue {
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

/// Per-CPU state definition (fits your percpu crate workflow).
///
/// You will implement `current()` using your percpu API:
/// e.g. EBR_CPU.current_ref_raw() / current_ref_mut_raw() but prefer returning `&EbrCpuRecord`
/// and keep internals atomic/locked to avoid &mut aliasing under IRQ nesting.
#[percpu::def_percpu]
static EBR_CPU: EbrCpuRecord = EbrCpuRecord {
    nesting: AtomicU32::new(0),
    active: AtomicBool::new(false),
    announced: AtomicU64::new(0),
    retire: [RetireQueue::new(), RetireQueue::new(), RetireQueue::new()],
};

/// Guard that pins the current CPU (preemption disabled) until dropped.
/// This mirrors the common "pin returns guard; drop unpins" pattern. :contentReference[oaicite:2]{index=2}
pub struct Guard {
    _np: NoPreempt,
    _cookie: GuardCookie,
}

/// Marker; can carry
