// modules/axsync/src/rcu.rs

use core::sync::atomic::{AtomicU64, Ordering};
use axhal::cpu::this_cpu_id;
use axhal::arch::{disable_irqs, enable_irqs};

// Global RCU epoch counter
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(1);

const MAX_CPU: usize = axhal::SMP_SMP_CORES;

static CPU_QS_EPOCH: [AtomicU64; MAX_CPU] = [const {AtomicU64{0}}; MAX_CPU];

fn this_cpu_state() -> &'static AtomicU64 {
    &CPU_QS_EPOCH[this_cpu_id()]
}