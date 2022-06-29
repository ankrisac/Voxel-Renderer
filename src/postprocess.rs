use crate::upscale::UpscaleTexture;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Vertex {
    pos: [f32; 3],
}
impl Vertex {
    fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        }
    }
}

fn upscale_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let label = Some("postprocess:upscale");
    let shader = crate::load_shader(device, std::path::Path::new("shaders/upscale.wgsl"), label);

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label,
        push_constant_ranges: &[],
        bind_group_layouts: &[&UpscaleTexture::layout(
            device,
            UpscaleTexture::USAGE_RENDER,
        )],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label,
        depth_stencil: None,
        layout: Some(&layout),

        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,

            clamp_depth: false,
            conservative: false,
        },
        vertex: wgpu::VertexState {
            entry_point: "main",
            module: &shader,
            buffers: &[Vertex::layout()],
        },
        fragment: Some(wgpu::FragmentState {
            entry_point: "main",
            module: &shader,
            targets: &[wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),

        multisample: wgpu::MultisampleState {
            mask: !0,
            count: 1,
            alpha_to_coverage_enabled: false,
        },
    })
}

fn bloom_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let label = Some("postprocess:bloom");

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label,
        bind_group_layouts: &[&UpscaleTexture::layout(
            device,
            UpscaleTexture::USAGE_COMPUTE,
        )],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label,
        module: &crate::load_shader(device, std::path::Path::new("shaders/bloom.wgsl"), label),
        layout: Some(&layout),
        entry_point: "main",
    })
}
pub struct Pipeline {
    bloom_pipeline: wgpu::ComputePipeline,
    upscale_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}
impl Pipeline {
    #[rustfmt::skip]

    const VERTICES: [Vertex; 4] = [
        Vertex { pos: [ 1.0,  1.0, 0.0] },
        Vertex { pos: [-1.0,  1.0, 0.0] },
        Vertex { pos: [-1.0, -1.0, 0.0] },
        Vertex { pos: [ 1.0, -1.0, 0.0] }
    ];
    const INDICES: [u16; 6] = [0, 1, 3, 1, 2, 3];

    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("postprocess:upscale"),
            usage: wgpu::BufferUsages::VERTEX,
            contents: bytemuck::bytes_of(&Self::VERTICES),
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("postprocess:upscale"),
            usage: wgpu::BufferUsages::INDEX,
            contents: bytemuck::bytes_of(&Self::INDICES),
        });

        let upscale_pipeline = upscale_pipeline(device);
        let bloom_pipeline = bloom_pipeline(device);

        Self {
            bloom_pipeline,
            upscale_pipeline,
            vertex_buffer,
            index_buffer,
        }
    }
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        upscale_texture: &UpscaleTexture,
        output_view: &wgpu::TextureView,
    ) {
        let (x, y) = upscale_texture.get_size();

        for _ in 0..4 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("post_process:bloom-pass"),
            });

            pass.set_pipeline(&self.bloom_pipeline);
            pass.set_bind_group(0, &upscale_texture.compute_bindgroup, &[]);
            pass.dispatch(x / 8 + 1, y / 8 + 1, 1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_process:upscale-pass"),
                depth_stencil_attachment: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: true,
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                    },
                }],
            });

            pass.set_pipeline(&self.upscale_pipeline);
            pass.set_bind_group(0, &upscale_texture.render_bindgroup, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..Self::INDICES.len() as _, 0, 0..1);
        }
    }
}
