pub struct RenderContext {
    surface_config: wgpu::SurfaceConfiguration,

    pub size: winit::dpi::PhysicalSize<u32>,
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl RenderContext {
    pub async fn new(window: &winit::window::Window) -> Self {
        let backends = wgpu::Backends::PRIMARY | wgpu::Backends::SECONDARY;
        let instance = wgpu::Instance::new(backends);

        let surface = unsafe { instance.create_surface(window) };
        let adapter: wgpu::Adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to create wgpu::Adapter!");

        let size = {

            let chunk_r = 8;
            let grid_r = 32;
            
            let chunk = chunk_r * chunk_r * chunk_r;
            let grid = grid_r * grid_r * grid_r;

            let voxel_bytes = std::mem::size_of::<u32>();
            let lod = 2;

            voxel_bytes * chunk * grid * lod * 4
        };

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    //    | wgpu::Features::TEXTURE_BINDING_ARRAY
                    //    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                    ,
                    limits: wgpu::Limits {
                        max_storage_buffer_binding_size: size as _,
                        ..Default::default()
                    },
                    label: Some("device"),
                },
                None,
            )
            .await
            .expect("Failed to create wgpu::Device!");

        let size = window.inner_size();

        let surface_config = wgpu::SurfaceConfiguration {
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        surface.configure(&device, &surface_config);

        Self {
            size,
            surface,
            surface_config,
            device,
            queue,
        }
    }
    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.size = size;
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;

        self.surface.configure(&self.device, &self.surface_config);
    }
}
