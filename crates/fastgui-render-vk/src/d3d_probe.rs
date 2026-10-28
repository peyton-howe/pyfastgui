//! Read a shared Direct3D 11 texture's real size and format (Windows).
//!
//! Vulkan accepts any `width`/`height` for an imported D3D texture, so a wrong claim makes the GPU
//! read past the allocation. This opens the NT handle on a D3D11 device created on the Vulkan
//! adapter (found by LUID) and reads the texture's own description. It also answers "is this handle
//! on the window's GPU": the open fails when it is not.

use ash::{vk, Instance};
use fastgui_core::SharedFormat;
use windows::core::Interface;
use windows::Win32::Foundation::{HANDLE, HMODULE};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM};
use windows::Win32::Graphics::Dxgi::*;

#[derive(Clone)]
pub(crate) struct D3dProbe {
    device: ID3D11Device1,
    luid: u64,
}

/// What a shared texture actually is.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SharedDesc {
    pub width: u32,
    pub height: u32,
    pub format: SharedFormat,
}

/// A probe on the adapter `pdevice` runs on.
pub(crate) unsafe fn for_device(instance: &Instance, pdevice: vk::PhysicalDevice) -> Result<D3dProbe, String> {
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    instance.get_physical_device_properties2(pdevice, &mut props);
    if id.device_luid_valid != vk::TRUE {
        return Err("the Vulkan device reports no adapter LUID, so a D3D texture cannot be checked".to_string());
    }
    D3dProbe::new(id.device_luid)
}

impl D3dProbe {
    pub(crate) fn new(luid: [u8; 8]) -> Result<Self, String> {
        let low = u32::from_le_bytes(luid[..4].try_into().unwrap());
        let high = i32::from_le_bytes(luid[4..].try_into().unwrap());
        let luid64 = u64::from(low) | (u64::from(high as u32) << 32);
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1().map_err(|e| format!("no DXGI factory: {e}"))?;
            let adapter = (0..)
                .map_while(|i| factory.EnumAdapters1(i).ok())
                .find(|a| a.GetDesc1().is_ok_and(|d| d.AdapterLuid.LowPart == low && d.AdapterLuid.HighPart == high))
                .ok_or_else(|| format!("no DXGI adapter with LUID {luid64:#x}"))?;
            let adapter: IDXGIAdapter = adapter.cast().map_err(|e| e.to_string())?;
            let mut device = None;
            D3D11CreateDevice(
                Some(&adapter),
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                None,
            )
            .map_err(|e| format!("could not create a D3D11 device on adapter {luid64:#x}: {e}"))?;
            let device: ID3D11Device1 = device
                .ok_or("no D3D11 device")?
                .cast()
                .map_err(|e| format!("the D3D11 device lacks ID3D11Device1: {e}"))?;
            Ok(Self { device, luid: luid64 })
        }
    }

    /// Open `handle` (an NT handle from `IDXGIResource1::CreateSharedHandle`) and describe the
    /// texture behind it. The texture is released again before this returns.
    pub(crate) fn describe(&self, handle: isize) -> Result<SharedDesc, String> {
        unsafe {
            let texture: ID3D11Texture2D = self.device.OpenSharedResource1(HANDLE(handle as *mut _)).map_err(|err| {
                format!(
                    "the handle could not be opened as a 2D texture on the window's GPU (adapter LUID {:#x}): {err}. Pass an NT handle from IDXGIResource1::CreateSharedHandle on a texture made by a D3D11 device on that adapter (fastgui.adapter_luid())",
                    self.luid
                )
            })?;
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            texture.GetDesc(&mut desc);
            let format = if desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM {
                SharedFormat::Rgba8
            } else if desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM {
                SharedFormat::Bgra8
            } else {
                return Err(format!(
                    "the texture's DXGI_FORMAT is {}; only R8G8B8A8_UNORM (28) and B8G8R8A8_UNORM (87) are imported",
                    desc.Format.0
                ));
            };
            if desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1 {
                return Err(format!(
                    "the texture has {} mip levels, {} array slices and {}x multisampling; only a single-level, single-slice, non-multisampled texture is imported",
                    desc.MipLevels, desc.ArraySize, desc.SampleDesc.Count
                ));
            }
            Ok(SharedDesc { width: desc.Width, height: desc.Height, format })
        }
    }
}
