//! Which GPU a window will use, before there is a window.

use ash::{vk, Entry};

/// The adapter LUID (`low | high << 32`) of the Vulkan device a window's renderer picks, so a D3D
/// producer can create its device on the same GPU. `None` off Windows, when no Vulkan 1.3 device
/// qualifies, or when the driver reports no LUID.
///
/// This applies the renderer's device filter (Vulkan 1.3, dynamic rendering, a graphics queue) and
/// takes the first match. The renderer also requires presentation support for its surface, which
/// every mainstream GPU has, so the two agree except on unusual multi-GPU setups.
pub fn adapter_luid() -> Option<u64> {
    unsafe {
        let entry = Entry::load().ok()?;
        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let instance = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app_info), None)
            .ok()?;
        let found = instance.enumerate_physical_devices().ok().and_then(|devices| {
            devices.into_iter().find(|&pdevice| {
                let props = instance.get_physical_device_properties(pdevice);
                if props.api_version < vk::make_api_version(0, 1, 3, 0) {
                    return false;
                }
                let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
                let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
                instance.get_physical_device_features2(pdevice, &mut features2);
                features13.dynamic_rendering == vk::TRUE
                    && instance
                        .get_physical_device_queue_family_properties(pdevice)
                        .iter()
                        .any(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            })
        });
        let luid = found.and_then(|pdevice| {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            instance.get_physical_device_properties2(pdevice, &mut props);
            (cfg!(windows) && id.device_luid_valid == vk::TRUE).then(|| {
                let low = u32::from_le_bytes(id.device_luid[..4].try_into().unwrap());
                let high = u32::from_le_bytes(id.device_luid[4..].try_into().unwrap());
                u64::from(low) | (u64::from(high) << 32)
            })
        });
        instance.destroy_instance(None);
        luid
    }
}
