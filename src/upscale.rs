#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UpscaleConfig {
    origin: [f32; 2],
    inv_height: f32,
    scale: f32,
}
impl UpscaleConfig {
    fn new(w: u32, h: u32, scale: u32) -> Self {
        let (w, h) = (w as f32, h as f32);

        Self {
            origin: [w / h, 1.0],
            inv_height: 2.0 / h,
            scale: scale as _,
        }
    }
    fn resize(&mut self, w: u32, h: u32, scale: u32) {
        let (w, h) = (w as f32, h as f32);

        self.origin = [w / h, 1.0];
        self.inv_height = 2.0 / h;
        self.scale = scale as _;
    }
}
pub struct UpscaleTexture {
    texture: wgpu::Texture,
    size: (u32, u32),

    config_buffer: wgpu::Buffer,
    config: UpscaleConfig,

    pub render_bindgroup: wgpu::BindGroup,
    pub compute_bindgroup: wgpu::BindGroup,
}
impl UpscaleTexture {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

    pub const USAGE_COMPUTE: (wgpu::ShaderStages, wgpu::StorageTextureAccess) = (
        wgpu::ShaderStages::COMPUTE,
        wgpu::StorageTextureAccess::ReadWrite,
    );
    pub const USAGE_RENDER: (wgpu::ShaderStages, wgpu::StorageTextureAccess) = (
        wgpu::ShaderStages::FRAGMENT,
        wgpu::StorageTextureAccess::ReadOnly,
    );

    pub fn new(device: &wgpu::Device, size: (u32, u32), scale: u32) -> Self {
        let width = size.0 / scale + 1;
        let height = size.1 / scale + 1;

        let size = (width, height);

        use wgpu::util::DeviceExt;
        let config = UpscaleConfig::new(width as _, height as _, scale as _);
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upscale-texture:buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(&config),
        });

        let texture = Self::make_texture(device, size);
        let view = Self::make_view(&texture);

        let input_bindgroup =
            Self::make_bindgroup(device, Self::USAGE_RENDER, &view, &config_buffer);
        let output_bindgroup =
            Self::make_bindgroup(device, Self::USAGE_COMPUTE, &view, &config_buffer);

        Self {
            texture,
            size,

            config,
            config_buffer,

            render_bindgroup: input_bindgroup,
            compute_bindgroup: output_bindgroup,
        }
    }
    fn make_view(texture: &wgpu::Texture) -> wgpu::TextureView {
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("upscale-texture:view"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        })
    }

    fn make_texture(device: &wgpu::Device, size: (u32, u32)) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("upscale-texture:texture-desc"),
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING,

            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },

            sample_count: 1,
            mip_level_count: 1,
        })
    }
    fn make_bindgroup(
        device: &wgpu::Device,
        usage: (wgpu::ShaderStages, wgpu::StorageTextureAccess),
        view: &wgpu::TextureView,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscale-texture:bindgroup"),
            layout: &Self::layout(device, usage),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn get_size(&self) -> (u32, u32) {
        self.size
    }

    pub fn resize(
        &mut self,
        context: &crate::context::RenderContext,
        size: (u32, u32),
        scale: u32,
    ) {
        let width = size.0 / scale + 1;
        let height = size.1 / scale + 1;
        self.size = (width, height);
        self.config.resize(width, height, scale);

        context
            .queue
            .write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&self.config));

        self.texture = Self::make_texture(&context.device, (width, height));
        let view = Self::make_view(&self.texture);

        self.render_bindgroup = Self::make_bindgroup(
            &context.device,
            Self::USAGE_RENDER,
            &view,
            &self.config_buffer,
        );
        self.compute_bindgroup = Self::make_bindgroup(
            &context.device,
            Self::USAGE_COMPUTE,
            &view,
            &self.config_buffer,
        );
    }

    pub fn layout(
        device: &wgpu::Device,
        usage: (wgpu::ShaderStages, wgpu::StorageTextureAccess),
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("upscale-texture:bindgroup-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: usage.0,
                    ty: wgpu::BindingType::StorageTexture {
                        access: usage.1,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        format: Self::FORMAT,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: usage.0,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        has_dynamic_offset: false,
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                },
            ],
        })
    }
}
