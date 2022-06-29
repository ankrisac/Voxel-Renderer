struct Mat4x4 {
    data: [[f32; 4]; 4],
}
impl Mat4x4 {
    fn diagonal(v: f32) -> Self {
        Self {
            data: [
                [v, 0.0, 0.0, 0.0],
                [0.0, v, 0.0, 0.0],
                [0.0, 0.0, v, 0.0],
                [0.0, 0.0, 0.0, v],
            ],
        }
    }
    fn rot(i: usize, j: usize, angle: f32) -> Self {
        let mut output = Self::diagonal(1.0);
        let (s, c) = angle.sin_cos();

        output.data[i][i] = c;
        output.data[j][i] = s;
        output.data[i][j] = -s;
        output.data[j][j] = c;
        output
    }
    fn transpose(self) -> Self {
        let mut output = Mat4x4::diagonal(0.0);
        for i in 0..4 {
            for j in 0..4 {
                output.data[i][j] = self.data[j][i];
            }
        }
        output
    }
    fn to_array(self) -> [[f32; 4]; 4] {
        self.data
    }
}
impl std::ops::Mul for Mat4x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut output = Mat4x4::diagonal(0.0);

        for i in 0..4 {
            for k in 0..4 {
                let mut sum = 0.0;
                for j in 0..4 {
                    sum += self.data[i][j] * rhs.data[j][k];
                }
                output.data[i][k] = sum;
            }
        }

        output
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct CameraData {
    transform: [[f32; 4]; 4],
}
pub struct Camera {
    angle_yz: f32,
    angle_xz: f32,

    buffer: wgpu::Buffer,
    pub bindgroup: wgpu::BindGroup,
}

impl Camera {
    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera:bindgroup:layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let buffer_data = CameraData {
            transform: Self::matrix_from(0.0, 0.0),
        };
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera:uniform"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(&buffer_data),
        });

        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera:bindgroup"),
            layout: &Self::layout(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Camera {
            angle_xz: 0.0,
            angle_yz: 0.0,
            buffer,
            bindgroup,
        }
    }

    pub fn update_from_controller(&mut self, controller: &ClientController) {
        self.angle_xz = controller.angle_xz;
        self.angle_yz = controller.angle_yz;
    }
    pub fn update(&mut self, context: &crate::context::RenderContext) {
        context.queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::bytes_of(&CameraData {
                transform: Self::matrix_from(self.angle_xz, self.angle_yz),
            }),
        );
    }
    fn matrix_from(angle_xz: f32, angle_yz: f32) -> [[f32; 4]; 4] {
        let rot_xz = Mat4x4::rot(0, 2, -angle_xz);
        let rot_yz = Mat4x4::rot(1, 2, angle_yz);

        (rot_xz * rot_yz).transpose().to_array()
    }
}

pub struct ClientController {
    pos: [f64; 3],
    vel: [f64; 3],
    acc: [f64; 3],
    hor_dir: [f64; 2],

    angle_yz: f32,
    angle_xz: f32,
}
impl ClientController {
    pub fn new() -> Self {
        Self {
            pos: [0.0, 0.0, 0.0],
            vel: [0.0, 0.0, 0.0],
            acc: [0.0, 0.0, 0.0],
            hor_dir: [0.0, 0.0],
            angle_yz: 0.0,
            angle_xz: 0.0,
        }
    }

    pub fn input_mouse(&mut self, delta: (f32, f32)) {
        let sensitivity = 0.005;
        self.angle_xz += delta.0 * sensitivity;
        self.angle_yz += delta.1 * sensitivity;
    }
    pub fn input_keyboard(&mut self, key: winit::event::VirtualKeyCode, pressed: bool) {
        let a: f64 = if pressed { 128.0 } else { 0.0 };

        use winit::event::VirtualKeyCode::*;
        match key {
            A => self.hor_dir[0] = -a,
            D => self.hor_dir[0] = a,
            
            S => self.hor_dir[1] = -a,
            W => self.hor_dir[1] = a,
            
            LShift => self.acc[1] = -a,
            Space => self.acc[1] = a,
            _ => {}
        }

        let (s, c) = (-self.angle_xz as f64).sin_cos();
        self.acc[0] = c * self.hor_dir[0] - s * self.hor_dir[1];
        self.acc[2] = s * self.hor_dir[0] + c * self.hor_dir[1];
    }
    pub fn update(&mut self, delta: f64) {
        let h = delta;

        for i in 0..3 {
            let dv = (self.acc[i] - 4.0 * self.vel[i]) * h;
            self.pos[i] += self.vel[i] * h + dv * h * 0.5;
            self.vel[i] += dv;
        }
    }

    pub fn get_pos(&self) -> [f64; 3] {
        self.pos
    }
}
