// modules/axtask/src/rcu.rs

pub use axsync::rcu::{
    Epoch, Guard, Retired, is_pinned, pin, poll, poll_budgeted, rcu_assign_ptr,
    rcu_deref_ptr, rcu_read_lock, rcu_read_unlock, retire, retire_ptr, synchronize,
};

const TICK_BUDGET: usize = 8;

/// Called at scheduler/timer safe points to amortize EBR reclamation.
///
/// Timer ticks should remain lightweight, so use a small budget.
pub fn rcu_poll_on_tick() -> bool {
    poll_budgeted(TICK_BUDGET)
}

/// Context switch hook: run a fuller poll to advance epochs and reclaim.
pub fn rcu_poll_on_context_switch() -> bool {
    poll()
}

/// Optional compatibility name mirroring legacy quiescent-state helpers.
pub fn quiescent_state_compat() -> bool {
    poll_budgeted(TICK_BUDGET)
}
