use core::sync::atomic::{AtomicUsize, Ordering};

/// A hook called when the current CPU reaches a quiescent state.
///
/// Stored as a raw function pointer to avoid crate dependency cycles.
static QS_HOOK: AtomicUsize = AtomicUsize::new(0);

/// Register a global quiescent-state hook.
///
/// Returns `true` if it is registered successfully, or `false` if a hook was
/// already registered.
pub fn register_quiescent_state_hook(hook: fn()) -> bool {
    let ptr = hook as usize;
    QS_HOOK
        .compare_exchange(0, ptr, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

#[inline(always)]
pub(crate) fn notify_quiescent_state() {
    let ptr = QS_HOOK.load(Ordering::Relaxed);
    if ptr != 0 {
        // Safety: only `fn()` pointers are stored here.
        let hook: fn() = unsafe { core::mem::transmute(ptr) };
        hook();
    }
}
