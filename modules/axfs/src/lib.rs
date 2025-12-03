#![no_std]
#![allow(clippy::new_ret_no_self)]
#![feature(maybe_uninit_slice)]

extern crate alloc;

#[macro_use]
extern crate log;

use axdriver::{AxBlockDevice, AxDeviceContainer, prelude::*};

#[cfg(feature = "fat")]
mod disk;
mod fs;

mod highlevel;
pub use highlevel::*;

pub fn init_filesystems(mut block_devs: AxDeviceContainer<AxBlockDevice>) {
    info!("Initialize filesystem subsystem...");

    let dev = block_devs.take_one().expect("No block device found!");
    info!("  use block device 0: {:?}", dev.device_name());

    let fs = fs::new_default(dev).expect("Failed to initialize filesystem");
    info!("  filesystem type: {:?}", fs.name());

    let mp = axfs_ng_vfs::Mountpoint::new_root(&fs);
    ROOT_FS_CONTEXT.call_once(|| FsContext::new(mp.root_location()));
}
