mod camera;
mod chunk;
mod context;
mod postprocess;
mod upscale;
struct WorldPipeline {
    size: (u32, u32),
    scale: u32,

    world: chunk::World,

    worldrender_pipeline: wgpu::ComputePipeline,
    postprocess_pipeline: postprocess::Pipeline,

    upscale_texture: upscale::UpscaleTexture,
    camera: camera::Camera,
}

fn load_shader<'a>(
    device: &wgpu::Device,
    path: &std::path::Path,
    label: Option<&'a str>,
) -> wgpu::ShaderModule {
    use std::io::prelude::*;
    let mut source = String::new();

    match std::fs::File::open(path) {
        Ok(mut file) => {
            if let Err(_) = file.read_to_string(&mut source) {
                panic!("Shader: Error reading {}", path.display())
            }
        }
        Err(_) => panic!("Shader: Error opening {}", path.display()),
    }
    device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: label,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

impl WorldPipeline {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute-pass:pipeline-layout"),
            push_constant_ranges: &[],
            bind_group_layouts: &[
                &upscale::UpscaleTexture::layout(device, upscale::UpscaleTexture::USAGE_COMPUTE),
                &camera::Camera::layout(device),
                &chunk::World::layout(device),
            ],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute-pass:compute-pipeline"),
            entry_point: "main",
            layout: Some(&compute_layout),
            module: &load_shader(
                device,
                std::path::Path::new("shaders/render.wgsl"),
                Some("compute-pass:shader"),
            ),
        });

        let scale = 3;

        let upscale_texture = upscale::UpscaleTexture::new(device, size, scale);
        let upscale_pipeline = postprocess::Pipeline::new(device);
        let camera = camera::Camera::new(device);

        let world = chunk::World::new(device);

        Self {
            size,
            scale,

            world,
            worldrender_pipeline: compute_pipeline,

            postprocess_pipeline: upscale_pipeline,
            upscale_texture,
            camera,
        }
    }
    fn resize(&mut self, context: &context::RenderContext, size: (u32, u32)) {
        self.size = size;
        self.upscale_texture.resize(context, self.size, self.scale);
    }
    fn render(&mut self, context: &context::RenderContext, view: &wgpu::TextureView) {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute-pipeline:command"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("world-render:compute-pass"),
            });

            pass.set_pipeline(&self.worldrender_pipeline);
            pass.set_bind_group(0, &self.upscale_texture.compute_bindgroup, &[]);
            pass.set_bind_group(1, &self.camera.bindgroup, &[]);
            pass.set_bind_group(2, &self.world.bindgroup, &[]);

            let (w, h) = (self.size.0 / self.scale, self.size.1 / self.scale);
            pass.dispatch(w / 32 + 1, h / 32 + 1, 1);
        }
        self.postprocess_pipeline
            .render(&mut encoder, &self.upscale_texture, view);

        context.queue.submit(std::iter::once(encoder.finish()));
    }
    fn update(&mut self, context: &context::RenderContext, controller: &camera::ClientController) {
        self.camera.update_from_controller(controller);
        self.camera.update(context);
        self.world.update(&context.queue, controller.get_pos());
    }
}

struct Engine {
    context: context::RenderContext,
    pipeline: WorldPipeline,
    controller: camera::ClientController,
    profiler: Profiler,
}
impl Engine {
    async fn new(window: &winit::window::Window) -> Self {
        let context = context::RenderContext::new(window).await;

        let size = context.size;
        let size = (size.width as u32, size.height as u32);
        let pipeline = WorldPipeline::new(&context.device, size);
        let controller = camera::ClientController::new();
        let profiler = Profiler::new();

        Self {
            context,
            pipeline,
            controller,
            profiler,
        }
    }
    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.context.resize(size);
        self.pipeline
            .resize(&self.context, (size.width as _, size.height as _));
    }

    fn render(&mut self) {
        match self.context.surface.get_current_frame() {
            Ok(surface_frame) => {
                let frame = surface_frame.output;
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                
                self.profiler.display();

                self.controller.update(self.profiler.get_delta());
                self.pipeline.update(&self.context, &self.controller);
                self.pipeline.render(&self.context, &view);
                self.profiler.new_frame();
            }
            Err(err) => {
                match err {
                    wgpu::SurfaceError::Lost => self.context.resize(self.context.size),
                    wgpu::SurfaceError::OutOfMemory => panic!("Render context out of memory!"),
                    _ => println!("Error: {:?}", err),
                }
            },
        }
    }
}

use std::collections::VecDeque;
struct Profiler {
    time_stamps: VecDeque<f64>,
    last_stamp: std::time::Instant,
}
impl Profiler {
    fn new() -> Self {
        let time_stamps = VecDeque::<f64>::new();
        let last_stamp = std::time::Instant::now();

        Self {
            time_stamps,
            last_stamp,
        }
    }
    fn new_frame(&mut self) {
        self.time_stamps
            .push_back(self.last_stamp.elapsed().as_secs_f64());

        if self.time_stamps.len() > 10 {
            self.time_stamps.pop_front();
        }

        self.last_stamp = std::time::Instant::now();
    }
    fn get_delta(&mut self) -> f64 {
        *self.time_stamps.get(0).unwrap_or(&0.0)
    }
    fn display(&mut self) {
        let mut sum: f64 = 0.0;
        let mut max: f64 = 0.0;
        let mut min: f64 = f64::INFINITY;
        for &val in &self.time_stamps {
            sum += val;
            max = max.max(val);
            min = min.min(val);
        }

        let average = sum / self.time_stamps.len() as f64;

        fn stat(label: &str, time: f64) {
            println!("{:9} {:10.9} | {:9.0}", label, time, 1.0 / time);
        }

        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

        println!("== Statistics ==");
        println!("{:9} {:11} | {:9} ", "", "s/frame", "FPS");
        stat("Average", average);
        stat("Min", min);
        stat("Max", max);
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("voxel-test")
        .with_maximized(true)
        .build(&event_loop)
        .unwrap();

    let mut state = futures::executor::block_on(Engine::new(&window));
    let mut cursor_grabbed = false;
    let _ = window.set_cursor_grab(cursor_grabbed);
    window.set_cursor_visible(!cursor_grabbed);

    event_loop.run(move |event, _event_loop, control_flow| {
        use winit::event::*;
        match event {
            Event::RedrawRequested(window_id) => {
                if window_id == window.id() {
                    state.render();
                }
            }
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta: (x, y) } => {
                    if cursor_grabbed {
                        state.controller.input_mouse((x as _, y as _));
                    }
                }
                _ => {}
            },
            Event::WindowEvent { window_id, event } => {
                if window_id == window.id() {
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = winit::event_loop::ControlFlow::Exit
                        }
                        WindowEvent::Resized(size) => state.resize(size),
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(*new_inner_size)
                        }
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => {
                                cursor_grabbed = !cursor_grabbed;
                                let _ = window.set_cursor_grab(cursor_grabbed);
                                window.set_cursor_visible(!cursor_grabbed);
                            }
                            KeyboardInput {
                                state: press_state,
                                virtual_keycode: Some(key),
                                ..
                            } => state
                                .controller
                                .input_keyboard(key, press_state == ElementState::Pressed),
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
            Event::MainEventsCleared => window.request_redraw(),
            _ => {}
        }
    });
}
