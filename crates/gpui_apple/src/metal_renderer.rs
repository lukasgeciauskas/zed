use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use gpui::{
    AtlasTextureId, Bounds, ContentMask, DevicePixels, LayerComposite, MAX_LAYER_MASK_STOPS,
    OffscreenLayer, PaintSurface, PaintedPath, PrimitiveBatch, ScaledPixels, Scene, SceneBatch,
    Size, quad_depth, quad_depth_scale, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::{CFRetained, CGSize};
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceDisplayP3};
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use objc2_foundation::{NSRange, NSString};
#[cfg(target_os = "macos")]
use objc2_metal::MTLCopyAllDevices;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction, MTLCreateSystemDefaultDevice,
    MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLDrawable, MTLGPUFamily,
    MTLLibrary, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage, MTLViewport,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use objc2_metal::{MTLBlitCommandEncoder, MTLOrigin, MTLRegion, MTLSize};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;

use std::{cell::Cell, mem, mem::MaybeUninit, ops::Range, ptr, slice, sync::Arc};

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub fn new_renderer(
    context: self::Context,
    transparent: bool,
    use_display_p3: bool,
) -> Result<Renderer> {
    MetalRenderer::new(context, transparent, use_display_p3)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<InstanceBuffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

struct InstanceBuffer {
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

// SAFETY: Each buffer has one CPU writer and returns to the mutex-protected pool only after GPU completion.
unsafe impl Send for InstanceBuffer {}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        unified_memory: bool,
    ) -> Result<InstanceBuffer> {
        if let Some(buffer) = self.buffers.pop() {
            return Ok(buffer);
        }
        let options = if unified_memory {
            MTLResourceOptions::StorageModeShared
                // Buffers are write only which can benefit from the combined cache
                // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                | MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            MTLResourceOptions::StorageModeManaged
        };
        let buffer = device
            .newBufferWithLength_options(self.buffer_size, options)
            .with_context(|| format!("allocating {}-byte instance buffer", self.buffer_size))?;
        Ok(InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        })
    }

    fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer)
        }
    }
}

enum DisplayLinkMode {
    Detached,
    Attached {
        drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    },
}

pub struct MetalRenderer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    layer: Option<Retained<CAMetalLayer>>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    display_link_mode: DisplayLinkMode,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    /// Whether the last scene drawn to the layer let native views show through it.
    shows_native_views: bool,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    path_tiles_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    shadows_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    quads_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    opaque_quads_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    underlines_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    monochrome_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    polychrome_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    surfaces_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    native_views_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Offscreen layers (Astryx patch R3): composites and the blur's passes.
    layer_composite_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    layer_blur_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Bound for a layer's painted mask when it has none.
    layer_mask_placeholder: Retained<ProtocolObject<dyn MTLTexture>>,
    /// Textures for offscreen layers, kept between frames.
    layer_pool: LayerTexturePool,
    depth_write_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    depth_test_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    depth_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
        use_display_p3: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = CAMetalLayer::new();
        Self::configure_layer(&layer, &device, transparent, use_display_p3);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view, such as
    /// the backing layer UIKit creates for a view whose `layerClass` is
    /// `CAMetalLayer`. The renderer retains the layer.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &CAMetalLayer,
        transparent: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = layer.retain();
        Self::configure_layer(&layer, &device, transparent, false);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    fn configure_layer(
        layer: &CAMetalLayer,
        device: &ProtocolObject<dyn MTLDevice>,
        transparent: bool,
        use_display_p3: bool,
    ) {
        layer.setDevice(Some(device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        if use_display_p3 {
            // SAFETY: The constant is extern, from Core Graphics
            let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
            if let Some(display_p3_color_space) = CGColorSpace::with_name(Some(display_p3_name)) {
                layer.setColorspace(Some(&display_p3_color_space));
            } else {
                log::error!("unable to create the Display P3 color space");
            }
        }
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.setOpaque(!transparent);
        layer.setMaximumDrawableCount(3);
        // Allow texture reading for visual tests and UI automation screenshots
        // (captures without ScreenCaptureKit). Only debug builds pay the
        // presentation cost, even when `test-support` is compiled in.
        #[cfg(all(feature = "test-support", debug_assertions))]
        layer.setFramebufferOnly(false);
        layer.setAllowsNextDrawableTimeout(false);
        layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        layer.setAutoresizingMask(
            objc2_quartz_core::CAAutoresizingMask::LayerWidthSizable
                | objc2_quartz_core::CAAutoresizingMask::LayerHeightSizable,
        );
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Result<Self> {
        let device = Self::create_device()?;
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    #[cfg(target_os = "macos")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(device) = MTLCopyAllDevices()
            .iter()
            .min_by_key(|device| (device.isRemovable(), !device.isLowPower()))
        {
            Ok(device)
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            Self::system_default_device()
        }
    }

    #[cfg(target_os = "ios")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        Self::system_default_device()
    }

    fn system_default_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        MTLCreateSystemDefaultDevice().context("accessing a compatible Metal device")
    }

    fn new_internal(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        layer: Option<Retained<CAMetalLayer>>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Result<Self> {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(SHADERS_SOURCE_FILE),
                Some(&objc2_metal::MTLCompileOptions::new()),
            )
            .context("building Metal library")?;
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .newLibraryWithData_error(&dispatch2::DispatchData::from_static_bytes(
                SHADERS_METALLIB,
            ))
            .context("loading Metal library")?;

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        // iOS does not support managed resources. Its simulator may report a
        // non-unified host GPU even though resources must still use shared storage.
        let is_unified_memory = cfg!(target_os = "ios") || device.hasUnifiedMemory();
        // Only Apple GPUs can create shared-storage textures on macOS.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supportsFamily(MTLGPUFamily::Apple1);

        let path_tiles_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "path_tiles",
            "path_tile_vertex",
            "path_tile_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        )?;
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        )?;
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        )?;
        let opaque_quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "opaque_quads",
            "opaque_quad_vertex",
            "opaque_quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        )?;
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        )?;
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        )?;
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        )?;
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
            true,
        )?;
        // Blending is off so that the transparent output replaces what was drawn beneath it.
        let native_views_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "native_views",
            "native_view_vertex",
            "native_view_fragment",
            MTLPixelFormat::BGRA8Unorm,
            false,
        )?;

        // Offscreen layers (Astryx patch R3): their textures hold premultiplied
        // colors, so composites blend them as such.
        let layer_composite_pipeline_state = build_layer_pipeline_state(
            &device,
            &library,
            "layer_composite",
            "layer_composite_vertex",
            "layer_composite_fragment",
            LayerPass::Composite,
        )?;
        let layer_blur_pipeline_state = build_layer_pipeline_state(
            &device,
            &library,
            "layer_blur",
            "layer_blur_vertex",
            "layer_blur_fragment",
            LayerPass::Blur,
        )?;
        let layer_mask_placeholder = {
            let descriptor = MTLTextureDescriptor::new();
            // SAFETY: A 1 by 1 texture is within every device's limits.
            unsafe {
                descriptor.setWidth(1);
                descriptor.setHeight(1);
            }
            descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            descriptor.setUsage(MTLTextureUsage::ShaderRead);
            descriptor.setStorageMode(MTLStorageMode::Private);
            device
                .newTextureWithDescriptor(&descriptor)
                .context("allocating the layer mask placeholder")?
        };

        let depth_write_state = build_depth_stencil_state(&device, true)?;
        let depth_test_state = build_depth_stencil_state(&device, false)?;

        let command_queue = device.newCommandQueue().context("creating command queue")?;
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "PlatformWindow::sprite_atlas requires Arc, not Send or Sync"
        )]
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        let mut cache = ptr::null_mut();
        // SAFETY: The output pointer is writable and both optional attribute dictionaries are absent.
        let status = unsafe {
            CVMetalTextureCache::create(None, None, &device, None, ptr::NonNull::from(&mut cache))
        };
        anyhow::ensure!(
            status == kCVReturnSuccess,
            "creating CoreVideo Metal texture cache: status {status}"
        );
        let cache = ptr::NonNull::new(cache).context("retrieving CoreVideo Metal texture cache")?;
        // SAFETY: The successful create call transfers ownership of this non-null cache.
        let core_video_texture_cache = unsafe { CFRetained::from_raw(cache) };

        Ok(Self {
            device,
            layer,
            presents_with_transaction: false,
            display_link_mode: DisplayLinkMode::Detached,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            shows_native_views: false,
            command_queue,
            path_tiles_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            opaque_quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            native_views_pipeline_state,
            layer_composite_pipeline_state,
            layer_blur_pipeline_state,
            layer_mask_placeholder,
            layer_pool: LayerTexturePool::default(),
            depth_write_state,
            depth_test_state,
            depth_texture: None,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        })
    }

    pub fn layer(&self) -> Option<&CAMetalLayer> {
        self.layer.as_deref()
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|layer| Retained::as_ptr(layer).cast_mut())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    /// Draw the next frame into `drawable` instead of asking the layer for one.
    ///
    /// A `CAMetalDisplayLink` hands out each frame's drawable early enough for
    /// the frame to reach the display on time; `nextDrawable` would instead
    /// wait for the compositor to release one. Pass `None` once the frame is
    /// done, so a drawable the frame did not use goes back to the layer.
    ///
    /// Returns the drawable this replaces: after a frame, `Some` means the
    /// frame neither drew nor presented.
    ///
    /// This is a setter rather than a guard scoped to one frame because the
    /// window releases its lock on the renderer while GPUI draws.
    pub fn set_frame_drawable(
        &mut self,
        drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    ) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable: current } => mem::replace(current, drawable),
            DisplayLinkMode::Detached => {
                // Only a live link hands out drawables, and a link is detached
                // only after it has been invalidated.
                debug_assert!(
                    drawable.is_none(),
                    "received a display link drawable while detached"
                );
                None
            }
        }
    }

    /// Record whether a `CAMetalDisplayLink` drives the layer. While one
    /// does, frames are drawn only into the drawables it hands out.
    pub fn set_display_link_attached(&mut self, attached: bool) {
        self.display_link_mode = if attached {
            DisplayLinkMode::Attached { drawable: None }
        } else {
            DisplayLinkMode::Detached
        };
    }

    pub fn is_display_link_attached(&self) -> bool {
        matches!(self.display_link_mode, DisplayLinkMode::Attached { .. })
    }

    pub fn presents_with_transaction(&self) -> bool {
        self.presents_with_transaction
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.setPresentsWithTransaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.setDrawableSize(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.setOpaque(self.opaque && !self.shows_native_views);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        // Native views sit beneath the layer, so it can only be opaque, and
        // eligible for direct-to-display presentation, while none are showing.
        let shows_native_views = !scene.native_views.is_empty();
        if shows_native_views != self.shows_native_views {
            self.shows_native_views = shows_native_views;
            layer.setOpaque(self.opaque && !shows_native_views);
        }
        let drawable = match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable } => {
                let Some(drawable) = drawable.take() else {
                    log::error!("skipped a frame drawn outside its display link's update");
                    return;
                };
                drawable
            }
            DisplayLinkMode::Detached => {
                let Some(drawable) = layer.nextDrawable() else {
                    log::error!(
                        "failed to retrieve next drawable, drawable size: {:?}",
                        layer.drawableSize()
                    );
                    return;
                };
                drawable
            }
        };
        // A display link's drawable was sized when the link handed it out,
        // which can be before a resize the layer has since taken.
        let texture = drawable.texture();
        let viewport_size: Size<DevicePixels> = size(
            (texture.width() as i32).into(),
            (texture.height() as i32).into(),
        );

        let command_buffer = match self.render_frame(scene, &drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.waitUntilScheduled();
            drawable.present();
        } else {
            command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
            command_buffer.commit();
        }
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        )?;
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue
            .commandBuffer()
            .context("creating command buffer")?;
        let mut surface_textures = Vec::new();
        let mut frame = FrameEncoder {
            command_buffer: &command_buffer,
            writer: &mut writer,
            surface_textures: &mut surface_textures,
        };
        self.layer_pool.begin_frame();
        // Offscreen layers first, each into its own texture, so the scene can
        // composite them (Astryx patch R3). Metal runs a command buffer's
        // passes in the order they were encoded.
        let result = match self.render_layers(scene, &mut frame) {
            Ok(layers) => {
                let result =
                    self.draw_frame_scene(scene, texture, viewport_size, &mut frame, &layers);
                self.release_layers(layers);
                result
            }
            Err(error) => Err(error),
        };
        self.layer_pool.end_frame();
        result?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let frame_resources = Cell::new(Some((writer.finish(), surface_textures)));
        let block = RcBlock::new(
            move |_: ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                if let Some((instance_buffer, _surface_textures)) = frame_resources.take() {
                    instance_buffer_pool.lock().release(instance_buffer);
                }
            },
        );
        // SAFETY: Metal copies the live block and invokes it once after this frame completes.
        // Its captures own the frame resources and use the pool mutex when returning the buffer.
        unsafe {
            command_buffer.addCompletedHandler(RcBlock::as_ptr(&block));
        }

        Ok(command_buffer)
    }

    /// Draws `scene` into the frame's target, compositing its offscreen
    /// layers from `layers`.
    fn draw_frame_scene(
        &mut self,
        scene: &Scene,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        frame: &mut FrameEncoder<'_>,
        layers: &[Option<RenderedLayer>],
    ) -> Result<()> {
        let instance_bindings = write_instances(scene, frame.writer).with_context(|| {
            format!(
                "writing instance data: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let depth_texture = self.depth_texture_matching(texture)?;
        let alpha = if self.opaque { 1. } else { 0. };
        let target = PassTarget {
            texture,
            depth_texture: &depth_texture,
            size: viewport_size,
            origin: [0.0, 0.0],
        };
        self.encode_scene(
            scene,
            &instance_bindings,
            frame,
            &target,
            MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha,
            },
            layers,
        )
    }

    /// Draws each offscreen layer of `scene` (and the layers inside them)
    /// into a pooled texture, blurred when its filter says so (Astryx patch
    /// R3). The result is indexed like `scene.offscreen_layers`.
    fn render_layers(
        &mut self,
        scene: &Scene,
        frame: &mut FrameEncoder<'_>,
    ) -> Result<Vec<Option<RenderedLayer>>> {
        let mut rendered = Vec::with_capacity(scene.offscreen_layers.len());
        for layer in &scene.offscreen_layers {
            match self.render_layer(layer, frame) {
                Ok(layer) => rendered.push(layer),
                Err(error) => {
                    self.release_layers(rendered);
                    return Err(error);
                }
            }
        }
        Ok(rendered)
    }

    fn render_layer(
        &mut self,
        layer: &OffscreenLayer,
        frame: &mut FrameEncoder<'_>,
    ) -> Result<Option<RenderedLayer>> {
        let width = layer.region.size.width.0;
        let height = layer.region.size.height.0;
        if width <= 0
            || height <= 0
            || width as u32 > MAX_LAYER_TEXTURE_SIZE
            || height as u32 > MAX_LAYER_TEXTURE_SIZE
        {
            return Ok(None);
        }
        let (width, height) = (width as u32, height as u32);
        let color = self.draw_layer_scene(&layer.content, layer, width, height, frame)?;
        let mask = match &layer.mask {
            Some(mask) => match self.draw_layer_scene(mask, layer, width, height, frame) {
                Ok(mask) => Some(mask),
                Err(error) => {
                    self.layer_pool.release(color);
                    return Err(error);
                }
            },
            None => None,
        };
        let rendered = RenderedLayer { color, mask };
        if layer.filter.blur > 0.0
            && let Err(error) =
                self.blur_layer(&rendered.color, layer, width, height, frame.command_buffer)
        {
            self.release_layers(vec![Some(rendered)]);
            return Err(error);
        }
        Ok(Some(rendered))
    }

    /// Draws `scene`, a layer's content or mask, into a pooled texture
    /// covering the layer's region.
    fn draw_layer_scene(
        &mut self,
        scene: &Scene,
        layer: &OffscreenLayer,
        width: u32,
        height: u32,
        frame: &mut FrameEncoder<'_>,
    ) -> Result<LayerTexture> {
        let children = self.render_layers(scene, frame)?;
        let texture = match self
            .layer_pool
            .acquire(&self.device, width, height, self.is_apple_gpu)
        {
            Ok(texture) => texture,
            Err(error) => {
                self.release_layers(children);
                return Err(error);
            }
        };
        let result = match write_instances(scene, frame.writer) {
            Ok(instance_bindings) => {
                let target = PassTarget {
                    texture: &texture.texture,
                    depth_texture: &texture.depth_texture,
                    size: size(DevicePixels(width as i32), DevicePixels(height as i32)),
                    origin: [
                        layer.region.origin.x.0 as f32,
                        layer.region.origin.y.0 as f32,
                    ],
                };
                self.encode_scene(
                    scene,
                    &instance_bindings,
                    frame,
                    &target,
                    MTLClearColor {
                        red: 0.,
                        green: 0.,
                        blue: 0.,
                        alpha: 0.,
                    },
                    &children,
                )
            }
            Err(error) => Err(error),
        };
        self.release_layers(children);
        match result {
            Ok(()) => Ok(texture),
            Err(error) => {
                self.layer_pool.release(texture);
                Err(error)
            }
        }
    }

    /// Blurs a layer's texture in place: horizontally into a scratch
    /// texture, then vertically back.
    fn blur_layer(
        &mut self,
        texture: &LayerTexture,
        layer: &OffscreenLayer,
        width: u32,
        height: u32,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        let scratch = self
            .layer_pool
            .acquire(&self.device, width, height, self.is_apple_gpu)?;
        let uniforms = |source: &LayerTexture, direction: [f32; 2]| LayerUniforms {
            region: [0.0, 0.0, width as f32, height as f32],
            texture_size: [source.width as f32, source.height as f32],
            blur_direction: direction,
            blur_sigma: layer.filter.blur,
            ..LayerUniforms::default()
        };
        let result = self
            .blur_pass(
                command_buffer,
                texture,
                &scratch,
                &uniforms(texture, [1.0, 0.0]),
            )
            .and_then(|()| {
                self.blur_pass(
                    command_buffer,
                    &scratch,
                    texture,
                    &uniforms(&scratch, [0.0, 1.0]),
                )
            });
        self.layer_pool.release(scratch);
        result
    }

    /// One direction of a layer's blur, from `source` into `destination`.
    fn blur_pass(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        source: &LayerTexture,
        destination: &LayerTexture,
        uniforms: &LayerUniforms,
    ) -> Result<()> {
        let render_pass_descriptor = MTLRenderPassDescriptor::new();
        // SAFETY: Index zero is within the render pass's color attachment array.
        let color_attachment = unsafe {
            render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
        };
        color_attachment.setTexture(Some(&destination.texture));
        color_attachment.setStoreAction(MTLStoreAction::Store);
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.,
            green: 0.,
            blue: 0.,
            alpha: 0.,
        });
        let command_encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .context("creating the layer blur's render command encoder")?;
        command_encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: uniforms.region[2] as f64,
            height: uniforms.region[3] as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        command_encoder.setRenderPipelineState(&self.layer_blur_pipeline_state);
        // SAFETY: Metal copies the uniforms, which match the shader's
        // `LayerUniforms`; the source texture outlives the command buffer's
        // use of it, since Metal retains encoded resources.
        unsafe {
            command_encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(uniforms).cast(),
                mem::size_of::<LayerUniforms>(),
                LayerInputIndex::Uniforms as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&source.texture),
                LayerInputIndex::Texture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                1,
                0,
            );
        }
        command_encoder.endEncoding();
        Ok(())
    }

    /// Composites a rendered layer where `composite` places it.
    fn draw_layer_composite(
        &self,
        composite: &LayerComposite,
        layer: &OffscreenLayer,
        rendered: &RenderedLayer,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        let uniforms =
            LayerUniforms::for_composite(composite, layer, &rendered.color, rendered.mask.as_ref());
        let mask = rendered
            .mask
            .as_ref()
            .map_or(&self.layer_mask_placeholder, |mask| &mask.texture);
        command_encoder.setRenderPipelineState(&self.layer_composite_pipeline_state);
        // SAFETY: Metal copies the uniforms, which match the shaders'
        // `LayerUniforms`, and retains the encoded textures.
        unsafe {
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&uniforms).cast(),
                mem::size_of_val(&uniforms),
                LayerInputIndex::Uniforms as usize,
            );
            command_encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&uniforms).cast(),
                mem::size_of_val(&uniforms),
                LayerInputIndex::Uniforms as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&rendered.color.texture),
                LayerInputIndex::Texture as usize,
            );
            command_encoder
                .setFragmentTexture_atIndex(Some(mask), LayerInputIndex::MaskTexture as usize);
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                1,
                0,
            );
        }
    }

    fn release_layers(&mut self, layers: Vec<Option<RenderedLayer>>) {
        for layer in layers.into_iter().flatten() {
            self.layer_pool.release(layer.color);
            if let Some(mask) = layer.mask {
                self.layer_pool.release(mask);
            }
        }
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawableSize();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        // CAMetalDisplayLink owns the drawable, so render into a texture we own.
        self.render_offscreen(scene, viewport_size)
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| self.render_offscreen(scene, size))
    }

    /// Renders a scene into a new offscreen texture of `size` and reads it
    /// back, without presenting anything.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    fn render_offscreen(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_offscreen: {:?}", size);
        }

        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
        unsafe {
            texture_descriptor.setWidth(size.width.0 as usize);
            texture_descriptor.setHeight(size.height.0 as usize);
        }
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        // Like the atlas, only Apple GPUs can create shared textures on macOS;
        // Intel Macs cannot, even with unified memory. iOS has no managed storage.
        let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
        texture_descriptor.setStorageMode(if uses_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let target_texture = self
            .device
            .newTextureWithDescriptor(&texture_descriptor)
            .context("allocating offscreen render target")?;

        let command_buffer = self.render_frame(scene, &target_texture, size)?;

        // Managed textures require an explicit blit synchronize before the CPU
        // can read back the rendered data. Without this, get_bytes returns
        // stale zeros.
        if !uses_shared_storage {
            let blit = command_buffer
                .blitCommandEncoder()
                .context("creating blit command encoder")?;
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        read_texture_to_image(&target_texture)
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as usize
                    || texture.height() != size.height.0 as usize
            });
            if needs_new_target {
                let texture_descriptor = MTLTextureDescriptor::new();
                unsafe {
                    texture_descriptor.setWidth(size.width.0 as usize);
                    texture_descriptor.setHeight(size.height.0 as usize);
                }
                texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor
                    .setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                texture_descriptor.setStorageMode(MTLStorageMode::Private);
                self.headless_render_target = Some(
                    self.device
                        .newTextureWithDescriptor(&texture_descriptor)
                        .context("allocating headless render target")?,
                );
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .context("retrieving headless render target")?;

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    /// Draws `scene` into `target` as one render pass of the frame,
    /// compositing its offscreen layers from `layers` (indexed like
    /// `scene.offscreen_layers`).
    fn encode_scene(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        frame: &mut FrameEncoder<'_>,
        target: &PassTarget<'_>,
        clear_color: MTLClearColor,
        layers: &[Option<RenderedLayer>],
    ) -> Result<()> {
        let viewport_size = target.size;
        let command_encoder = new_command_encoder_for_texture(
            frame.command_buffer,
            target.texture,
            target.depth_texture,
            viewport_size,
            clear_color,
        )?;
        let depth_scale = quad_depth_scale(scene.quads.len());
        // SAFETY: Metal copies the values, which match every shader's
        // `GlobalInputIndex_ViewportSize`, `GlobalInputIndex_DepthScale` and
        // `GlobalInputIndex_Origin` arguments.
        unsafe {
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                GlobalInputIndex::ViewportSize as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&depth_scale).cast(),
                mem::size_of_val(&depth_scale),
                GlobalInputIndex::DepthScale as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&target.origin).cast(),
                mem::size_of_val(&target.origin),
                GlobalInputIndex::Origin as usize,
            );
            command_encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&target.origin).cast(),
                mem::size_of_val(&target.origin),
                GlobalInputIndex::Origin as usize,
            );
        }

        self.draw_opaque_quads(scene.quads.len(), instance_bindings, &command_encoder);
        command_encoder.setDepthStencilState(Some(&self.depth_test_state));

        let mut quad_cursor: u32 = 0;
        let mut path_tile_cursor = 0;
        for batch in scene.batches_with_layers() {
            // Quad shaders assign each quad its own depth.
            // In every other batch type, the depth is fixed to the position of the quad cursor.
            let (znear, zfar) =
                if matches!(&batch, SceneBatch::Primitives(PrimitiveBatch::Quads(_))) {
                    (0.0, 1.0)
                } else {
                    let depth = quad_depth(quad_cursor, depth_scale) as f64;
                    (depth, depth)
                };
            command_encoder.setViewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: i32::from(viewport_size.width) as f64,
                height: i32::from(viewport_size.height) as f64,
                znear,
                zfar,
            });

            let batch = match batch {
                SceneBatch::Primitives(batch) => batch,
                SceneBatch::Layers(range) => {
                    for composite in &scene.layer_composites[range] {
                        let index = composite.layer as usize;
                        if let (Some(layer), Some(Some(rendered))) =
                            (scene.offscreen_layers.get(index), layers.get(index))
                        {
                            self.draw_layer_composite(composite, layer, rendered, &command_encoder);
                        }
                    }
                    continue;
                }
            };

            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, &command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    quad_cursor += range.len() as u32;
                    self.draw_quads(range, instance_bindings, &command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let tiles_start = path_tile_cursor;
                    path_tile_cursor += scene.paths[range]
                        .iter()
                        .map(PaintedPath::tile_count)
                        .sum::<usize>();
                    self.draw_path_tiles(
                        tiles_start..path_tile_cursor,
                        instance_bindings,
                        &command_encoder,
                    )
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, &command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        &command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        &command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => {
                    if let Err(error) = self.draw_surfaces(
                        &scene.surfaces[range.clone()],
                        range.start,
                        instance_bindings,
                        &command_encoder,
                        frame.surface_textures,
                    ) {
                        command_encoder.endEncoding();
                        return Err(error);
                    }
                }
                PrimitiveBatch::NativeViews(range) => {
                    self.draw_native_views(range, instance_bindings, &command_encoder)
                }
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.endEncoding();

        Ok(())
    }

    fn depth_texture_matching(
        &mut self,
        target: &ProtocolObject<dyn MTLTexture>,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>> {
        if let Some(depth_texture) = &self.depth_texture
            && depth_texture.width() == target.width()
            && depth_texture.height() == target.height()
        {
            return Ok(depth_texture.clone());
        }
        let descriptor = MTLTextureDescriptor::new();
        // SAFETY: The size is the size of an existing render target, which the device supports.
        unsafe {
            descriptor.setWidth(target.width());
            descriptor.setHeight(target.height());
        }
        descriptor.setPixelFormat(MTLPixelFormat::Depth32Float);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        // The depth buffer never leaves its render pass, so Apple GPUs can keep
        // it in tile memory without allocating any backing store.
        descriptor.setStorageMode(if self.is_apple_gpu {
            MTLStorageMode::Memoryless
        } else {
            MTLStorageMode::Private
        });
        let depth_texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .context("allocating depth texture")?;
        self.depth_texture = Some(depth_texture.clone());
        Ok(depth_texture)
    }

    fn draw_opaque_quads(
        &self,
        quad_count: usize,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if quad_count == 0 {
            return;
        }
        let quad_count_value = quad_count as u32;

        command_encoder.setRenderPipelineState(&self.opaque_quads_pipeline_state);
        command_encoder.setDepthStencilState(Some(&self.depth_write_state));
        // SAFETY: The scene's quads and the copied count match the shader layouts and slots.
        // Metal retains the buffer, and writer.finish publishes its writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&quad_count_value).cast(),
                mem::size_of_val(&quad_count_value),
                QuadInputIndex::QuadCount as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                quad_count,
                0,
            );
        }
    }

    fn draw_native_views(
        &self,
        native_views: Range<usize>,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if native_views.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.native_views_pipeline_state);
        // SAFETY: The SurfaceBounds instances match the shader layout and slot.
        // Metal retains the buffer, and writer.finish publishes its writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.native_views.buffer),
                instance_bindings.native_views.offset,
                SurfaceInputIndex::Surfaces as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                native_views.len(),
                native_views.start,
            );
        }
    }

    fn draw_path_tiles(
        &self,
        tiles: Range<usize>,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if tiles.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.path_tiles_pipeline_state);
        // SAFETY: The scene's tile range and its curves and paints match the uploaded shader layouts and slots.
        // Metal retains the buffers, and writer.finish publishes their writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.path_tiles.buffer),
                instance_bindings.path_tiles.offset,
                PathTileInputIndex::Tiles as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.path_paints.buffer),
                instance_bindings.path_paints.offset,
                PathTileInputIndex::Paints as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.path_tiles.buffer),
                instance_bindings.path_tiles.offset,
                PathTileInputIndex::Tiles as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.path_curves.buffer),
                instance_bindings.path_curves.offset,
                PathTileInputIndex::Curves as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.path_paints.buffer),
                instance_bindings.path_paints.offset,
                PathTileInputIndex::Paints as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                tiles.len(),
                tiles.start,
            );
        }
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.shadows_pipeline_state);
        // SAFETY: The scene's shadow range matches the uploaded shader layouts and slots.
        // Metal retains the buffers, and writer.finish publishes their writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                shadows.len(),
                shadows.start,
            );
        }
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.quads_pipeline_state);
        // SAFETY: The scene's quad range matches the uploaded shader layouts and slots.
        // Metal retains the buffers, and writer.finish publishes their writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                quads.len(),
                quads.start,
            );
        }
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.underlines_pipeline_state);
        // SAFETY: The scene's underline range matches the uploaded shader layouts and slots.
        // Metal retains the buffers, and writer.finish publishes their writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                underlines.len(),
                underlines.start,
            );
        }
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.setRenderPipelineState(&self.monochrome_sprites_pipeline_state);
        // SAFETY: The scene's sprite range and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.monochrome_sprites.buffer),
                instance_bindings.monochrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.monochrome_sprites.buffer),
                instance_bindings.monochrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                sprites.len(),
                sprites.start,
            );
        }
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.setRenderPipelineState(&self.polychrome_sprites_pipeline_state);
        // SAFETY: The scene's sprite range and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                sprites.len(),
                sprites.start,
            );
        }
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
    ) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }

        command_encoder.setRenderPipelineState(&self.surfaces_pipeline_state);
        // SAFETY: The SurfaceBounds instances match the shader layout and slot.
        // Metal retains the buffers, and writer.finish publishes CPU writes before submission.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.surfaces.buffer),
                instance_bindings.surfaces.offset,
                SurfaceInputIndex::Surfaces as usize,
            );
        }

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = &surface.image_buffer;
            let texture_size = size(
                DevicePixels::from(CVPixelBufferGetWidth(image_buffer) as i32),
                DevicePixels::from(CVPixelBufferGetHeight(image_buffer) as i32),
            );

            let pixel_format = CVPixelBufferGetPixelFormatType(image_buffer);
            anyhow::ensure!(
                pixel_format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                "unsupported surface pixel format: {pixel_format:#x}"
            );

            let plane_texture = |plane, format| -> Result<CFRetained<CVMetalTexture>> {
                let width = CVPixelBufferGetWidthOfPlane(image_buffer, plane);
                let height = CVPixelBufferGetHeightOfPlane(image_buffer, plane);
                let mut texture = ptr::null_mut();
                // SAFETY: The buffer has two planes matching these formats and dimensions, and the output pointer is writable.
                let status = unsafe {
                    CVMetalTextureCache::create_texture_from_image(
                        None,
                        &self.core_video_texture_cache,
                        image_buffer,
                        None,
                        format,
                        width,
                        height,
                        plane,
                        ptr::NonNull::from(&mut texture),
                    )
                };
                anyhow::ensure!(
                    status == kCVReturnSuccess,
                    "creating CoreVideo Metal texture for plane {plane}: status {status}"
                );
                let texture =
                    ptr::NonNull::new(texture).context("retrieving CoreVideo Metal texture")?;
                // SAFETY: The successful create call transfers ownership of this non-null texture.
                Ok(unsafe { CFRetained::from_raw(texture) })
            };
            let y_texture = plane_texture(0, MTLPixelFormat::R8Unorm)?;
            let cb_cr_texture = plane_texture(1, MTLPixelFormat::RG8Unorm)?;
            let y_metal_texture =
                CVMetalTextureGetTexture(&y_texture).context("retrieving luma Metal texture")?;
            let cb_cr_metal_texture = CVMetalTextureGetTexture(&cb_cr_texture)
                .context("retrieving chroma Metal texture")?;

            // SAFETY: The size is copied from a live value, and both planes match their shader texture slots.
            // The frame completion handler retains the CoreVideo wrappers until GPU reads finish.
            unsafe {
                command_encoder.setVertexBytes_length_atIndex(
                    ptr::NonNull::from(&texture_size).cast(),
                    mem::size_of_val(&texture_size),
                    SurfaceInputIndex::TextureSize as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&y_metal_texture),
                    SurfaceInputIndex::YTexture as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&cb_cr_metal_texture),
                    SurfaceInputIndex::CbCrTexture as usize,
                );
            }
            surface_textures.extend([y_texture, cb_cr_texture]);

            // SAFETY: This draws one instance within the uploaded scene's surface range.
            unsafe {
                command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                    1,
                    first_surface + index,
                );
            }
        }
        Ok(())
    }
}

fn new_command_encoder_for_texture(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    texture: &ProtocolObject<dyn MTLTexture>,
    depth_texture: &ProtocolObject<dyn MTLTexture>,
    viewport_size: Size<DevicePixels>,
    clear_color: MTLClearColor,
) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
    let render_pass_descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let color_attachment = unsafe {
        render_pass_descriptor
            .colorAttachments()
            .objectAtIndexedSubscript(0)
    };
    color_attachment.setTexture(Some(texture));
    color_attachment.setStoreAction(MTLStoreAction::Store);
    color_attachment.setLoadAction(MTLLoadAction::Clear);
    color_attachment.setClearColor(clear_color);

    let depth_attachment = render_pass_descriptor.depthAttachment();
    depth_attachment.setTexture(Some(depth_texture));
    depth_attachment.setLoadAction(MTLLoadAction::Clear);
    depth_attachment.setClearDepth(0.0);
    // The depth buffer is only needed within this pass.
    depth_attachment.setStoreAction(MTLStoreAction::DontCare);

    let command_encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
        .context("creating render command encoder")?;
    command_encoder.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    Ok(command_encoder)
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &ProtocolObject<dyn MTLTexture>) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        },
    };
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            ptr::NonNull::from(pixels.as_mut_slice()).cast(),
            bytes_per_row,
            region,
            0,
        );
    }

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("creating RGBA image from pixel data")
}

fn build_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
    blending: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(blending);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn build_depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
    write_depth: bool,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
    let descriptor = MTLDepthStencilDescriptor::new();
    descriptor.setDepthCompareFunction(MTLCompareFunction::Greater);
    descriptor.setDepthWriteEnabled(write_depth);
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .context("creating depth stencil state")
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    path_tiles: InstanceBinding,
    path_curves: InstanceBinding,
    path_paints: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    surfaces: InstanceBinding,
    native_views: InstanceBinding,
}

fn write_instances(scene: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBindings> {
    let mut path_paints = Vec::with_capacity(scene.paths.len());
    let mut path_curves = Vec::new();
    let mut path_tiles = Vec::new();
    for path in &scene.paths {
        path.append_to(&mut path_paints, &mut path_curves, &mut path_tiles);
    }
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        path_tiles: writer.write(&path_tiles)?,
        path_curves: writer.write(&path_curves)?,
        path_paints: writer.write(&path_paints)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
        }))?,
        native_views: writer.write_iter(scene.native_views.iter().map(|native_view| {
            SurfaceBounds {
                bounds: native_view.bounds,
                content_mask: native_view.content_mask,
            }
        }))?,
    })
}

struct InstanceBufferWriter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Result<Self> {
        let current = pool.lock().acquire(device, unified_memory)?;
        Ok(Self {
            device: device.retain(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        })
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // SAFETY: The aligned reservation lies within this frame's exclusively owned buffer and does not overlap prior writes.
        let values = unsafe {
            let start = self
                .current
                .metal_buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        // SAFETY: The disjoint source and destination hold the same number of T values, and the destination stays MaybeUninit.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory)?;
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: *written,
                });
            }
            if offset > 0 {
                current.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: offset,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

/// Vertex buffer slots bound once per render pass and shared by every
/// pipeline. Each primitive's own slots start after these.
#[repr(C)]
enum GlobalInputIndex {
    ViewportSize = 0,
    DepthScale = 1,
    /// Where the render target's top-left pixel is in the window's device
    /// pixels: zero for the window, a layer's region for an offscreen layer
    /// (Astryx patch R3). Fragment shaders that read their position bind it
    /// too.
    Origin = 2,
}

#[repr(C)]
enum ShadowInputIndex {
    Shadows = 3,
}

#[repr(C)]
enum QuadInputIndex {
    Quads = 3,
    QuadCount = 4,
}

#[repr(C)]
enum UnderlineInputIndex {
    Underlines = 3,
}

#[repr(C)]
enum SpriteInputIndex {
    Sprites = 3,
    AtlasTextureSize = 4,
    AtlasTexture = 5,
}

#[repr(C)]
enum SurfaceInputIndex {
    Surfaces = 3,
    TextureSize = 4,
    YTexture = 5,
    CbCrTexture = 6,
}

#[repr(C)]
enum PathTileInputIndex {
    Tiles = 3,
    Curves = 4,
    Paints = 5,
}

/// The slots of an offscreen layer's composite and blur passes (Astryx patch
/// R3): its uniforms in the buffer table, its texture and painted mask in the
/// texture table.
#[repr(C)]
enum LayerInputIndex {
    Uniforms = 3,
    Texture = 0,
    MaskTexture = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

// --- offscreen layers (Astryx patch R3) ---

/// The largest side of an offscreen layer's texture: Metal's limit on every
/// Mac GPU family GPUI supports.
const MAX_LAYER_TEXTURE_SIZE: u32 = 16384;

/// What a frame's passes are encoded into: its command buffer, its instance
/// data and the surfaces it keeps alive.
struct FrameEncoder<'a> {
    command_buffer: &'a ProtocolObject<dyn MTLCommandBuffer>,
    writer: &'a mut InstanceBufferWriter,
    surface_textures: &'a mut Vec<CFRetained<CVMetalTexture>>,
}

/// What a pass draws into: the frame's target or a layer's texture.
struct PassTarget<'a> {
    texture: &'a ProtocolObject<dyn MTLTexture>,
    depth_texture: &'a ProtocolObject<dyn MTLTexture>,
    /// The viewport, from the target's top-left.
    size: Size<DevicePixels>,
    /// Where the target's top-left pixel is in the window's device pixels.
    origin: [f32; 2],
}

/// A texture an offscreen layer is drawn into, with its depth buffer.
struct LayerTexture {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    depth_texture: Retained<ProtocolObject<dyn MTLTexture>>,
    width: u32,
    height: u32,
    last_used_frame: u64,
}

/// A layer drawn this frame, ready to composite.
struct RenderedLayer {
    color: LayerTexture,
    mask: Option<LayerTexture>,
}

/// Layer textures kept between frames, so a layer that animates reuses its
/// texture instead of creating one every frame. Sizes are rounded up so
/// layers that grow a little still fit; textures unused for a while are
/// dropped. A texture handed back while the GPU still reads it is safe to
/// draw into again: Metal orders the passes that use a tracked resource.
#[derive(Default)]
struct LayerTexturePool {
    free: Vec<LayerTexture>,
    frame: u64,
}

/// How many frames an unused layer texture is kept.
const LAYER_TEXTURE_IDLE_FRAMES: u64 = 120;
/// Texture sizes are multiples of this.
const LAYER_TEXTURE_GRANULARITY: u32 = 64;

impl LayerTexturePool {
    fn begin_frame(&mut self) {
        self.frame += 1;
    }

    fn end_frame(&mut self) {
        let frame = self.frame;
        self.free
            .retain(|texture| frame - texture.last_used_frame <= LAYER_TEXTURE_IDLE_FRAMES);
    }

    /// A texture at least `width` by `height`, from the pool when one fits
    /// without wasting much of it.
    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        width: u32,
        height: u32,
        is_apple_gpu: bool,
    ) -> Result<LayerTexture> {
        let round = |size: u32| {
            size.next_multiple_of(LAYER_TEXTURE_GRANULARITY)
                .min(MAX_LAYER_TEXTURE_SIZE)
                .max(size)
        };
        let (rounded_width, rounded_height) = (round(width), round(height));
        let area = |texture: &LayerTexture| texture.width as u64 * texture.height as u64;
        // Fits, without more than twice the rounded size on either side.
        let best = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, texture)| {
                texture.width >= width
                    && texture.height >= height
                    && texture.width <= rounded_width * 2
                    && texture.height <= rounded_height * 2
            })
            .min_by_key(|(_, texture)| area(texture))
            .map(|(index, _)| index);
        if let Some(index) = best {
            return Ok(self.free.swap_remove(index));
        }
        let (width, height) = (rounded_width, rounded_height);
        let descriptor = MTLTextureDescriptor::new();
        // SAFETY: The size is positive and at most MAX_LAYER_TEXTURE_SIZE,
        // which every Mac GPU family supports.
        unsafe {
            descriptor.setWidth(width as usize);
            descriptor.setHeight(height as usize);
        }
        descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Private);
        let texture = device
            .newTextureWithDescriptor(&descriptor)
            .context("allocating an offscreen layer's texture")?;
        descriptor.setPixelFormat(MTLPixelFormat::Depth32Float);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        // The depth buffer never leaves its pass, so Apple GPUs keep it in
        // tile memory.
        descriptor.setStorageMode(if is_apple_gpu {
            MTLStorageMode::Memoryless
        } else {
            MTLStorageMode::Private
        });
        let depth_texture = device
            .newTextureWithDescriptor(&descriptor)
            .context("allocating an offscreen layer's depth texture")?;
        Ok(LayerTexture {
            texture,
            depth_texture,
            width,
            height,
            last_used_frame: 0,
        })
    }

    fn release(&mut self, mut texture: LayerTexture) {
        texture.last_used_frame = self.frame;
        self.free.push(texture);
    }
}

const _: () = assert!(MAX_LAYER_MASK_STOPS * 2 == 16);

/// The `LayerUniforms` of `shaders.metal`: one composite or blur pass's
/// parameters. Plain arrays, so the generated header lays them out alike.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct LayerUniforms {
    /// The texture's region in its content's device pixels: origin, then size.
    pub region: [f32; 4],
    /// The size of the whole texture, which can be larger than the region.
    pub texture_size: [f32; 2],
    pub opacity: f32,
    pub brightness: f32,
    /// Maps the region to where it is drawn: the rotation and scale, row by
    /// row, then the translation.
    pub matrix: [f32; 4],
    pub translation: [f32; 2],
    pub grayscale: f32,
    pub mask_stop_count: u32,
    /// The composite's clip in the target's device pixels: origin, then size.
    pub clip: [f32; 4],
    /// The gradient mask's parameter, `dot(position, xy) + z`; the last
    /// value is 1 when a painted mask texture is sampled.
    pub mask_gradient: [f32; 4],
    /// The gradient mask's `(parameter, alpha)` stops, in pairs.
    pub mask_stops: [f32; 16],
    /// The blur's step, (1, 0) or (0, 1), and its standard deviation.
    pub blur_direction: [f32; 2],
    pub blur_sigma: f32,
    pub _pad: f32,
    /// The painted mask's texture size; it covers the same region.
    pub mask_texture_size: [f32; 2],
}

impl Default for LayerUniforms {
    fn default() -> Self {
        Self {
            region: [0.0; 4],
            texture_size: [1.0; 2],
            opacity: 1.0,
            brightness: 1.0,
            matrix: [1.0, 0.0, 0.0, 1.0],
            translation: [0.0; 2],
            grayscale: 0.0,
            mask_stop_count: 0,
            clip: [0.0; 4],
            mask_gradient: [0.0; 4],
            mask_stops: [0.0; 16],
            blur_direction: [0.0; 2],
            blur_sigma: 0.0,
            _pad: 0.0,
            mask_texture_size: [1.0; 2],
        }
    }
}

impl LayerUniforms {
    fn for_composite(
        composite: &LayerComposite,
        layer: &OffscreenLayer,
        texture: &LayerTexture,
        mask: Option<&LayerTexture>,
    ) -> Self {
        let filter = &layer.filter;
        let [[a, b], [c, d]] = composite.transformation.rotation_scale;
        let mut mask_stops = [0.0; 16];
        for (index, stop) in filter.mask_stops.iter().enumerate() {
            mask_stops[index * 2] = stop[0];
            mask_stops[index * 2 + 1] = stop[1];
        }
        let clip = composite.content_mask.bounds;
        Self {
            region: [
                composite.bounds.origin.x.0,
                composite.bounds.origin.y.0,
                composite.bounds.size.width.0,
                composite.bounds.size.height.0,
            ],
            texture_size: [texture.width as f32, texture.height as f32],
            opacity: filter.opacity,
            brightness: filter.brightness,
            matrix: [a, b, c, d],
            translation: composite.transformation.translation,
            grayscale: filter.grayscale,
            mask_stop_count: filter.mask_stop_count,
            clip: [
                clip.origin.x.0,
                clip.origin.y.0,
                clip.size.width.0,
                clip.size.height.0,
            ],
            mask_gradient: [
                filter.mask_gradient[0],
                filter.mask_gradient[1],
                filter.mask_gradient[2],
                if mask.is_some() { 1.0 } else { 0.0 },
            ],
            mask_stops,
            mask_texture_size: mask
                .map_or([1.0; 2], |mask| [mask.width as f32, mask.height as f32]),
            ..Self::default()
        }
    }
}

/// Which of a layer's passes a pipeline draws.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerPass {
    /// Into a scene's pass: premultiplied blending, depth tested like the
    /// primitives around it.
    Composite,
    /// Into a scratch texture with no depth buffer, replacing its pixels.
    Blur,
}

fn build_layer_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pass: LayerPass,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    let composite = pass == LayerPass::Composite;
    color_attachment.setBlendingEnabled(composite);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    if composite {
        descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    }

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Result<Self> {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool)?;
        Ok(Self { renderer })
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::Point;
    use objc2_core_foundation::{CFDictionary, CFType};
    use objc2_core_video::{
        CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey,
    };
    use objc2_metal::MTLCommandBufferStatus;

    #[test]
    fn render_to_image_rejects_zero_sized_layer() -> Result<()> {
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        for (width, height) in [(0, 0), (0, 64), (64, 0)] {
            renderer.update_drawable_size(size(DevicePixels(width), DevicePixels(height)));
            assert!(renderer.render_to_image(&Scene::default()).is_err());
        }
        Ok(())
    }

    #[test]
    fn display_link_mode_consumes_and_clears_drawables() -> Result<()> {
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        renderer.update_drawable_size(size(DevicePixels(64), DevicePixels(64)));
        let drawable = renderer
            .layer()
            .and_then(|layer| layer.nextDrawable())
            .context("window renderer must supply a drawable")?;

        renderer.set_display_link_attached(true);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: Some(_) }
        ));
        renderer.draw(&Scene::default());
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        // A frame that drew consumed its drawable; one that didn't hands it back.
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        assert!(renderer.set_frame_drawable(None).is_some());

        renderer.set_frame_drawable(Some(drawable));
        renderer.set_display_link_attached(false);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Detached
        ));
        Ok(())
    }

    #[test]
    fn render_core_video_surface() {
        let surface_properties_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        let attributes = CFDictionary::from_slices(
            &[surface_properties_key],
            &[&*CFDictionary::<CFType, CFType>::empty()],
        );
        let mut image_buffer = ptr::null_mut();
        assert_eq!(
            unsafe {
                CVPixelBufferCreate(
                    None,
                    16,
                    16,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    Some(attributes.as_opaque()),
                    ptr::NonNull::from(&mut image_buffer),
                )
            },
            kCVReturnSuccess
        );
        let image_buffer = ptr::NonNull::new(image_buffer).unwrap();
        let image_buffer = unsafe { CFRetained::from_raw(image_buffer) };
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
            for plane in 0..2 {
                let bytes = CVPixelBufferGetBaseAddressOfPlane(&image_buffer, plane).cast::<u8>();
                let length = CVPixelBufferGetBytesPerRowOfPlane(&image_buffer, plane)
                    * CVPixelBufferGetHeightOfPlane(&image_buffer, plane);
                slice::from_raw_parts_mut(bytes, length).fill(128);
            }
            assert_eq!(
                CVPixelBufferUnlockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
        }

        let bounds = Bounds {
            origin: Point::default(),
            size: size(ScaledPixels(16.0), ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            image_buffer: image_buffer.clone(),
        });
        drop(image_buffer);
        scene.finish();

        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())))
                .unwrap();
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(16), DevicePixels(16)))
            .unwrap();
        for image::Rgba([red, green, blue, alpha]) in image.pixels() {
            assert!(red.abs_diff(128) <= 2);
            assert!(green.abs_diff(128) <= 2);
            assert!(blue.abs_diff(128) <= 2);
            assert_eq!(*alpha, 255);
        }

        let descriptor = MTLTextureDescriptor::new();
        unsafe {
            descriptor.setWidth(16);
            descriptor.setHeight(16);
        }
        descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        let uses_shared_storage = cfg!(target_os = "ios") || renderer.is_apple_gpu;
        descriptor.setStorageMode(if uses_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let target_texture = renderer
            .device
            .newTextureWithDescriptor(&descriptor)
            .unwrap();
        let command_buffer = renderer
            .render_frame(
                &scene,
                &target_texture,
                size(DevicePixels(16), DevicePixels(16)),
            )
            .unwrap();
        if !uses_shared_storage {
            let blit = command_buffer.blitCommandEncoder().unwrap();
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }
        drop(scene);
        drop(renderer);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(read_texture_to_image(&target_texture).unwrap(), image);
    }
}

/// Offscreen layers (Astryx patch R3) drawn by the headless Metal renderer and
/// read back: the same cases as the DirectX and wgpu renderers' tests.
#[cfg(test)]
mod layer_tests {
    use super::*;
    use gpui::{
        BorderStyle, Corners, Edges, Hsla, LayerFilter, LayerParams, PlatformHeadlessRenderer,
        Point, Quad, Radians, TransformationMatrix,
    };

    const SIZE: i32 = 32;

    fn device_size() -> Size<DevicePixels> {
        size(DevicePixels(SIZE), DevicePixels(SIZE))
    }

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: ScaledPixels(x),
                y: ScaledPixels(y),
            },
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn quad(x: f32, y: f32, width: f32, height: f32, color: Hsla) -> Quad {
        let bounds = bounds(x, y, width, height);
        Quad {
            order: 0,
            border_style: BorderStyle::Solid,
            bounds,
            content_mask: ContentMask { bounds },
            background: color.into(),
            border_color: color,
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    fn params(filter: LayerFilter) -> LayerParams {
        let window = bounds(0., 0., SIZE as f32, SIZE as f32);
        let mut params = LayerParams::new(ContentMask { bounds: window }, window);
        params.filter = filter;
        params
    }

    fn render(scene: &mut Scene) -> image::RgbaImage {
        scene.finish();
        let mut renderer = MetalHeadlessRenderer::new().expect("a Metal device");
        renderer
            .render_scene_to_image(scene, device_size())
            .expect("the scene renders")
    }

    #[track_caller]
    fn assert_pixel(image: &image::RgbaImage, x: u32, y: u32, expected: [u8; 4]) {
        let actual = image.get_pixel(x, y).0;
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.abs_diff(expected) <= 4),
            "pixel ({x}, {y}) was {actual:?}, expected {expected:?}"
        );
    }

    #[test]
    fn a_layer_fades_as_one_group() {
        let mut scene = Scene::default();
        scene.push_offscreen(params(LayerFilter {
            opacity: 0.5,
            ..LayerFilter::default()
        }));
        scene.insert_primitive(quad(4., 4., 16., 16., gpui::red()));
        // Drawn over the red one: in the group only blue shows there.
        scene.insert_primitive(quad(12., 4., 16., 16., gpui::blue()));
        scene.pop_offscreen();
        let image = render(&mut scene);
        assert_pixel(&image, 8, 8, [128, 0, 0, 255]);
        assert_pixel(&image, 16, 8, [0, 0, 128, 255]);
        assert_pixel(&image, 24, 8, [0, 0, 128, 255]);
        assert_pixel(&image, 2, 2, [0, 0, 0, 255]);
    }

    #[test]
    fn a_layer_rotates_its_content() {
        let mut scene = Scene::default();
        // A bar from (14, 4) to (18, 28), turned a quarter around the center.
        let rotation = TransformationMatrix::unit()
            .translate(Point::new(ScaledPixels(16.), ScaledPixels(16.)))
            .rotate(Radians(std::f32::consts::FRAC_PI_2))
            .translate(Point::new(ScaledPixels(-16.), ScaledPixels(-16.)));
        let mut rotated = params(LayerFilter::default());
        rotated.transformation = rotation;
        scene.push_offscreen(rotated);
        scene.insert_primitive(quad(14., 4., 4., 24., gpui::red()));
        scene.pop_offscreen();
        let image = render(&mut scene);
        assert_pixel(&image, 6, 16, [255, 0, 0, 255]);
        assert_pixel(&image, 26, 16, [255, 0, 0, 255]);
        assert_pixel(&image, 16, 6, [0, 0, 0, 255]);
        assert_pixel(&image, 16, 26, [0, 0, 0, 255]);
    }

    #[test]
    fn filters_change_the_layers_colors() {
        let mut scene = Scene::default();
        scene.push_offscreen(params(LayerFilter {
            grayscale: 1.0,
            ..LayerFilter::default()
        }));
        scene.insert_primitive(quad(0., 0., 16., 32., gpui::red()));
        scene.pop_offscreen();
        scene.push_offscreen(params(LayerFilter {
            brightness: 0.5,
            ..LayerFilter::default()
        }));
        scene.insert_primitive(quad(16., 0., 16., 32., gpui::blue()));
        scene.pop_offscreen();
        let image = render(&mut scene);
        // CSS grayscale weighs red by 0.2126.
        assert_pixel(&image, 8, 16, [54, 54, 54, 255]);
        assert_pixel(&image, 24, 16, [0, 0, 128, 255]);
    }

    #[test]
    fn a_gradient_mask_fades_the_layer() {
        let mut filter = LayerFilter::default();
        // Alpha from 0 at x = 0 to 1 at x = 32.
        filter.mask_gradient = [1. / 32., 0., 0.];
        filter.mask_stops[0] = [0., 0.];
        filter.mask_stops[1] = [1., 1.];
        filter.mask_stop_count = 2;
        let mut scene = Scene::default();
        scene.push_offscreen(params(filter));
        scene.insert_primitive(quad(0., 0., 32., 32., gpui::white()));
        scene.pop_offscreen();
        let image = render(&mut scene);
        let at = |x: u32| image.get_pixel(x, 16).0[0] as f32 / 255.;
        // Fragments are sampled at pixel centers.
        assert!((at(8) - 8.5 / 32.).abs() < 0.03, "{}", at(8));
        assert!((at(24) - 24.5 / 32.).abs() < 0.03, "{}", at(24));
    }

    #[test]
    fn a_painted_mask_hides_what_it_leaves_out() {
        let mut scene = Scene::default();
        scene.push_offscreen(params(LayerFilter::default()));
        scene.insert_primitive(quad(0., 0., 32., 32., gpui::red()));
        scene.push_offscreen_mask();
        scene.insert_primitive(quad(0., 0., 16., 32., gpui::black()));
        scene.pop_offscreen_mask();
        scene.pop_offscreen();
        let image = render(&mut scene);
        assert_pixel(&image, 8, 16, [255, 0, 0, 255]);
        assert_pixel(&image, 24, 16, [0, 0, 0, 255]);
    }

    #[test]
    fn a_blur_spreads_the_layer() {
        let mut scene = Scene::default();
        scene.push_offscreen(params(LayerFilter {
            blur: 2.0,
            ..LayerFilter::default()
        }));
        scene.insert_primitive(quad(12., 12., 8., 8., gpui::white()));
        scene.pop_offscreen();
        let image = render(&mut scene);
        let red = |x: u32, y: u32| image.get_pixel(x, y).0[0];
        // Light reaches past the edge, and the center loses some.
        assert!(red(10, 16) > 20, "{}", red(10, 16));
        assert!(red(16, 16) < 250 && red(16, 16) > 150, "{}", red(16, 16));
        assert!(red(2, 2) < 3, "{}", red(2, 2));
        // Symmetric left and right.
        assert!(red(10, 16).abs_diff(red(21, 16)) <= 4);
    }

    #[test]
    fn nested_layers_multiply() {
        let mut scene = Scene::default();
        let half = LayerFilter {
            opacity: 0.5,
            ..LayerFilter::default()
        };
        scene.push_offscreen(params(half));
        scene.push_offscreen(params(half));
        scene.insert_primitive(quad(0., 0., 32., 32., gpui::white()));
        scene.pop_offscreen();
        scene.pop_offscreen();
        let image = render(&mut scene);
        assert_pixel(&image, 16, 16, [64, 64, 64, 255]);
    }

    #[test]
    fn layers_are_drawn_in_paint_order() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0., 0., 32., 32., gpui::red()));
        scene.push_offscreen(params(LayerFilter::default()));
        scene.insert_primitive(quad(0., 0., 16., 32., gpui::blue()));
        scene.pop_offscreen();
        scene.insert_primitive(quad(8., 0., 4., 32., gpui::green()));
        let image = render(&mut scene);
        assert_pixel(&image, 4, 16, [0, 0, 255, 255]);
        assert_pixel(&image, 10, 16, [0, 128, 0, 255]);
        assert_pixel(&image, 24, 16, [255, 0, 0, 255]);
    }

    #[test]
    fn a_layer_draws_where_its_region_sits() {
        // A layer away from the window's corner: its texture's origin is the
        // region's, which every shader subtracts.
        let mut scene = Scene::default();
        scene.push_offscreen(params(LayerFilter {
            opacity: 0.5,
            ..LayerFilter::default()
        }));
        scene.insert_primitive(quad(20., 20., 8., 8., gpui::white()));
        scene.pop_offscreen();
        let image = render(&mut scene);
        assert_pixel(&image, 24, 24, [128, 128, 128, 255]);
        assert_pixel(&image, 18, 24, [0, 0, 0, 255]);
        assert_pixel(&image, 24, 18, [0, 0, 0, 255]);
    }

    #[test]
    fn layer_textures_are_reused_across_frames() {
        let mut renderer = MetalHeadlessRenderer::new().expect("a Metal device");
        for _ in 0..3 {
            let mut scene = Scene::default();
            scene.push_offscreen(params(LayerFilter {
                blur: 1.0,
                ..LayerFilter::default()
            }));
            scene.insert_primitive(quad(4., 4., 8., 8., gpui::white()));
            scene.pop_offscreen();
            scene.finish();
            renderer
                .render_scene_to_image(&scene, device_size())
                .expect("the scene renders");
        }
        // The layer's texture and the blur's scratch one.
        assert_eq!(renderer.renderer.layer_pool.free.len(), 2);
    }
}
