"""Show a Direct3D 11 texture in a Viewport with no CPU copy (Windows).

A producer thread owns a D3D11 device on the same GPU as the window, renders into a small ring of
shared textures, and hands each one to `Viewport.submit_d3d11`. fastgui imports a texture the first
time it sees it and re-acquires it on every later submit, so the steady state is one barrier and a
sample. This is plain ctypes (no pywin32, no comtypes); a real producer would be a camera or video
decoder that already owns D3D11 textures.

    python d3d11_viewport.py                 # RGBA texture ring, event-query sync
    python d3d11_viewport.py --bgra          # BGRA, as DXGI desktop duplication produces
    python d3d11_viewport.py --fence         # sync with a shared ID3D11Fence instead
    python d3d11_viewport.py --seconds 10    # close the window itself after 10 seconds
    python d3d11_viewport.py --color 255,64,0   # a fixed color, to check channel order by eye

Run it with the Vulkan validation layer installed and any validation message is printed by the
renderer on stderr. Exits non-zero when the render thread reports anything but "ok".
"""

import argparse
import ctypes as C
import ctypes.wintypes as W
import math
import sys
import threading
import time
import uuid

import fastgui as fg

WIDTH, HEIGHT = 640, 360
TITLE = "fastgui d3d11 viewport"

HRESULT = C.c_long
DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM = 28, 87
D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_RENDER_TARGET = 0x8, 0x20
D3D11_RESOURCE_MISC_SHARED, D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x2, 0x800
D3D11_QUERY_EVENT, D3D11_FENCE_FLAG_SHARED = 0, 2
SHARED_ACCESS = 0x80000001  # DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE


def guid(text: str):
    return (C.c_ubyte * 16).from_buffer_copy(uuid.UUID(text).bytes_le)


IID_IDXGIFactory1 = guid("770aae78-f26f-4dba-a829-253c83d1b387")
IID_IDXGIResource1 = guid("30961379-4609-4a41-998e-54fe567ee0c1")
IID_ID3D11Device5 = guid("8ffde202-a0e7-45df-9e01-e837801b5ea0")
IID_ID3D11DeviceContext4 = guid("917600da-f58c-4c33-98d8-3e15b390fa24")
IID_ID3D11Fence = guid("affde9d1-1df7-4bb7-8a34-0f46251dab80")


class LUID(C.Structure):
    _fields_ = [("low", C.c_uint32), ("high", C.c_int32)]


class DXGI_ADAPTER_DESC1(C.Structure):
    _fields_ = [
        ("Description", C.c_wchar * 128),
        ("VendorId", C.c_uint), ("DeviceId", C.c_uint), ("SubSysId", C.c_uint), ("Revision", C.c_uint),
        ("DedicatedVideoMemory", C.c_size_t), ("DedicatedSystemMemory", C.c_size_t),
        ("SharedSystemMemory", C.c_size_t),
        ("AdapterLuid", LUID), ("Flags", C.c_uint),
    ]


class DXGI_SAMPLE_DESC(C.Structure):
    _fields_ = [("Count", C.c_uint), ("Quality", C.c_uint)]


class D3D11_TEXTURE2D_DESC(C.Structure):
    _fields_ = [
        ("Width", C.c_uint), ("Height", C.c_uint), ("MipLevels", C.c_uint), ("ArraySize", C.c_uint),
        ("Format", C.c_uint), ("SampleDesc", DXGI_SAMPLE_DESC), ("Usage", C.c_uint),
        ("BindFlags", C.c_uint), ("CPUAccessFlags", C.c_uint), ("MiscFlags", C.c_uint),
    ]


class D3D11_QUERY_DESC(C.Structure):
    _fields_ = [("Query", C.c_uint), ("MiscFlags", C.c_uint)]


def check(hr: int, what: str) -> None:
    if hr < 0:
        raise OSError(f"{what} failed: HRESULT {hr & 0xFFFFFFFF:#010x}")


class Com:
    """A raw COM pointer. `call(index, ...)` invokes vtable slot `index` (IUnknown is 0 to 2)."""

    def __init__(self, ptr: int):
        self.ptr = C.c_void_p(ptr)

    def call(self, index, restype, argtypes, *args):
        vtable = C.cast(self.ptr, C.POINTER(C.POINTER(C.c_void_p)))[0]
        function = C.WINFUNCTYPE(restype, C.c_void_p, *argtypes)(vtable[index])
        return function(self.ptr, *args)

    def query(self, iid) -> "Com":
        out = C.c_void_p()
        check(self.call(0, HRESULT, (C.c_void_p, C.c_void_p), C.addressof(iid), C.byref(out)), "QueryInterface")
        return Com(out.value)

    def release(self) -> None:
        self.call(2, C.c_ulong, ())


def find_adapter(luid: int | None) -> Com | None:
    """The DXGI adapter with this LUID (`fastgui.adapter_luid()`), or None for the default."""
    if luid is None:
        return None
    factory_ptr = C.c_void_p()
    create_factory = C.windll.dxgi.CreateDXGIFactory1
    create_factory.argtypes, create_factory.restype = [C.c_void_p, C.c_void_p], HRESULT
    check(create_factory(C.addressof(IID_IDXGIFactory1), C.addressof(factory_ptr)), "CreateDXGIFactory1")
    factory = Com(factory_ptr.value)
    index = 0
    while True:
        adapter_ptr = C.c_void_p()
        if factory.call(12, HRESULT, (C.c_uint, C.POINTER(C.c_void_p)), index, C.byref(adapter_ptr)) < 0:
            raise OSError(f"no DXGI adapter has LUID {luid:#x}")
        adapter = Com(adapter_ptr.value)
        desc = DXGI_ADAPTER_DESC1()
        check(adapter.call(10, HRESULT, (C.c_void_p,), C.byref(desc)), "GetDesc1")
        if (desc.AdapterLuid.low | (desc.AdapterLuid.high & 0xFFFFFFFF) << 32) == luid:
            print(f"producer adapter: {desc.Description}")
            return adapter
        adapter.release()
        index += 1


class Producer:
    """A D3D11 device that renders into a ring of shared textures."""

    def __init__(self, luid, bgra: bool, count: int, use_fence: bool, color=None):
        self.color = color
        adapter = find_adapter(luid)
        device, context = C.c_void_p(), C.c_void_p()
        create = C.windll.d3d11.D3D11CreateDevice
        create.argtypes = [C.c_void_p, C.c_uint, C.c_void_p, C.c_uint, C.c_void_p, C.c_uint, C.c_uint,
                           C.POINTER(C.c_void_p), C.c_void_p, C.POINTER(C.c_void_p)]
        # D3D_DRIVER_TYPE_UNKNOWN (an explicit adapter) or HARDWARE (the default one).
        check(create(adapter.ptr if adapter else None, 0 if adapter else 1, None, 0, None, 0, 7,
                     C.byref(device), None, C.byref(context)), "D3D11CreateDevice")
        self.device, self.context = Com(device.value), Com(context.value)
        self.format = DXGI_FORMAT_B8G8R8A8_UNORM if bgra else DXGI_FORMAT_R8G8B8A8_UNORM
        self.textures = [self._make_texture() for _ in range(count)]
        self.query = self._make_query()
        self.fence = self._make_fence() if use_fence else None

    def _make_texture(self):
        desc = D3D11_TEXTURE2D_DESC(
            WIDTH, HEIGHT, 1, 1, self.format, DXGI_SAMPLE_DESC(1, 0), 0,
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 0,
            D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
        )
        texture = C.c_void_p()
        check(self.device.call(5, HRESULT, (C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)),
                               C.byref(desc), None, C.byref(texture)), "CreateTexture2D")
        view = C.c_void_p()
        check(self.device.call(9, HRESULT, (C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)),
                               texture, None, C.byref(view)), "CreateRenderTargetView")
        handle = C.c_void_p()
        resource = Com(texture.value).query(IID_IDXGIResource1)
        check(resource.call(13, HRESULT, (C.c_void_p, C.c_uint, C.c_wchar_p, C.POINTER(C.c_void_p)),
                            None, SHARED_ACCESS, None, C.byref(handle)), "CreateSharedHandle")
        # (render target view, NT handle for fastgui, last fence value fastgui will signal for it)
        return [view, handle.value, 0]

    def _make_query(self) -> Com:
        desc, query = D3D11_QUERY_DESC(D3D11_QUERY_EVENT, 0), C.c_void_p()
        check(self.device.call(24, HRESULT, (C.c_void_p, C.POINTER(C.c_void_p)), C.byref(desc), C.byref(query)),
              "CreateQuery")
        return Com(query.value)

    def _make_fence(self):
        device5 = self.device.query(IID_ID3D11Device5)
        fence = C.c_void_p()
        check(device5.call(68, HRESULT, (C.c_uint64, C.c_uint, C.c_void_p, C.POINTER(C.c_void_p)),
                           0, D3D11_FENCE_FLAG_SHARED, C.addressof(IID_ID3D11Fence), C.byref(fence)), "CreateFence")
        fence = Com(fence.value)
        handle = C.c_void_p()
        check(fence.call(7, HRESULT, (C.c_void_p, C.c_uint, C.c_wchar_p, C.POINTER(C.c_void_p)),
                         None, 0x10000000, None, C.byref(handle)), "ID3D11Fence.CreateSharedHandle")
        return fence, handle.value, self.context.query(IID_ID3D11DeviceContext4)

    def render(self, slot: int, t: float) -> None:
        r, g, b = self.color or (0.5 + 0.5 * math.sin(t * k + p) for k, p in ((1.0, 0.0), (1.7, 2.0), (2.3, 4.0)))
        color = (C.c_float * 4)(r, g, b, 1.0)
        self.context.call(50, None, (C.c_void_p, C.c_void_p), self.textures[slot][0], C.addressof(color))

    def finish(self) -> None:
        """Flush and wait until the GPU has run everything queued (the no-fence handshake)."""
        self.context.call(28, None, (C.c_void_p,), self.query.ptr)
        self.context.call(111, None, ())
        done = C.c_uint(0)
        while self.context.call(29, HRESULT, (C.c_void_p, C.c_void_p, C.c_uint, C.c_uint),
                                self.query.ptr, C.byref(done), 4, 0) != 0 or not done.value:
            time.sleep(0.0005)

    def completed(self) -> int:
        return self.fence[0].call(8, C.c_uint64, ())

    def signal(self, value: int) -> None:
        check(self.fence[2].call(147, HRESULT, (C.c_void_p, C.c_uint64), self.fence[0].ptr, value), "Signal")
        self.context.call(111, None, ())


def produce(viewport: fg.Viewport, producer: Producer, stop: threading.Event, stats: dict) -> None:
    start = time.monotonic()
    frame = 0
    fmt = "bgra8" if producer.format == DXGI_FORMAT_B8G8R8A8_UNORM else None  # None: detected from the texture
    while not stop.is_set():
        slot = frame % len(producer.textures)
        _, handle, released = producer.textures[slot]
        t = time.monotonic() - start
        if producer.fence:
            fence, fence_handle, _ = producer.fence
            # Wait until fastgui has finished sampling this texture's previous frame.
            while producer.completed() < released and not stop.is_set():
                time.sleep(0.0005)
            producer.render(slot, t)
            wait_value, signal_value = 2 * frame + 1, 2 * frame + 2
            producer.signal(wait_value)
            viewport.submit_d3d11(handle, WIDTH, HEIGHT, format=fmt, fence=fence_handle,
                                  wait_value=wait_value, signal_value=signal_value)
            producer.textures[slot][2] = signal_value
        else:
            producer.render(slot, t)
            producer.finish()
            viewport.submit_d3d11(handle, WIDTH, HEIGHT, format=fmt)
        frame += 1
        stats["frames"] = frame
        stats["status"] = viewport.d3d11_status
        if frame % 30 == 0:
            print(f"frame {frame}: {stats['status']}")
        stop.wait(1 / 30)


def close_window_later(seconds: float) -> None:
    def close() -> None:
        time.sleep(seconds)
        hwnd = C.windll.user32.FindWindowW(None, TITLE)
        if hwnd:
            C.windll.user32.PostMessageW(W.HWND(hwnd), 0x0010, 0, 0)  # WM_CLOSE

    threading.Thread(target=close, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bgra", action="store_true", help="BGRA textures instead of RGBA")
    parser.add_argument("--fence", action="store_true", help="sync with a shared ID3D11Fence")
    parser.add_argument("--textures", type=int, default=3, help="ring size (default 3)")
    parser.add_argument("--seconds", type=float, help="close the window after this long")
    parser.add_argument("--color", help="a fixed R,G,B (0-255) instead of the animation")
    args = parser.parse_args()
    if sys.platform != "win32":
        print("submit_d3d11 is Windows-only")
        return 2

    luid = fg.adapter_luid()
    print(f"window adapter LUID: {luid:#x}" if luid is not None else "window adapter LUID: unknown")
    window = fg.Window(title=TITLE, width=WIDTH, height=HEIGHT)
    viewport = fg.Viewport(fit="contain")
    window.set_viewport(viewport)
    color = tuple(int(c) / 255 for c in args.color.split(",")) if args.color else None
    producer = Producer(luid, args.bgra, args.textures, args.fence, color)

    stop, stats = threading.Event(), {"frames": 0, "status": "unused"}
    thread = threading.Thread(target=produce, args=(viewport, producer, stop, stats), daemon=True)
    thread.start()
    if args.seconds:
        close_window_later(args.seconds)
    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=2.0)
    print(f"submitted {stats['frames']} frames; last status: {stats['status']}")
    return 0 if stats["frames"] and stats["status"].startswith("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
