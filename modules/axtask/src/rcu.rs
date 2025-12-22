// modules/axtask/src/rcu.rs
#![allow(dead_code)]

pub use axsync::rcu::{
    Epoch, Guard,
    pin, is_pinned,
    rcu_read_lock, rcu_read_unlock,
    rcu_assign_ptr, rcu_deref_ptr,
    Retired, retire, retire_ptr,
    poll, synchronize,
};

/// Called at scheduler/timer safe points to amortize EBR reclamation.
///
/// Suggested call sites (pick what fits your kernel):
/// - timer tick handler end
/// - context switch path
/// - runqueue operations batch end
pub fn rcu_poll_on_tick() -> bool {
    poll()
}

pub fn rcu_poll_on_context_switch() -> bool {
    poll()
}

/// Optional compatibility name: if the old QSBR code called something like `quiescent_state()`,
/// you can keep the symbol but change semantics to EBR poll.
/// (This helps incremental refactor.)
pub fn quiescent_state_compat() -> bool {
    poll()
}
