macro_rules! GPU {
    ($i:item) => {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        $i
    }
}

GPU! { 
    struct PaletteItem {
        color: [f32; 3],
        lum: f32,
    }
}

const MAX_LOD: usize = 8;

impl PaletteItem {
    fn color(color: [f32; 3]) -> Self {
        Self { color, lum: 0.0 }
    }
    fn air() -> Self {
        Self::color([0.0; 3])
    }

    fn light(color: [f32; 3], lum: f32) -> Self {
        Self { color, lum }
    }
    fn metal(color: [f32; 3], spec: f32) -> Self {
        Self { color, lum: -spec }
    }
}

struct Palette {
    data: Vec<PaletteItem>,
    buffer: wgpu::Buffer,
}
impl Palette {
    const DEBUG: [[f32; 3]; 3] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    const LIGHT: [[f32; 3]; 4] = [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    const METAL: [[f32; 3]; 4] = [
        [1.0, 1.0, 1.0],
        [0.702, 0.243, 0.24],
        [0.161, 0.671, 0.16],
        [0.239, 0.259, 0.67],
    ];

    const RANGE_DEBUG: Range<u32> = 1..4;
    const RANGE_LIGHT: Range<u32> = 4..8;
    const RANGE_METAL: Range<u32> = 8..12;
    const RANGE_GENERAL: Range<u32> = 12..(12 + 32u32.pow(3));

    #[rustfmt::skip]
    fn new(device: &wgpu::Device) -> Self {
        let mut data = Vec::new();

        data.push(PaletteItem::air());

        for color in Self::DEBUG {
            data.push(PaletteItem::color(color));
        }
        for color in Self::LIGHT {
            data.push(PaletteItem::light(color, 50.0));
        }
        for color in Self::METAL {
            data.push(PaletteItem::metal(color, 0.05));
        }        

        let n = 32;
        let ds = 0.005;

        let col = [
            (0.05, 0.26, 0.05),
        ];
        for (r, g, b) in col {
            let mut i = r;
            for _ in 0..n {
                let mut j = g;
                for _ in 0..n {
                    let mut k = b;
                    for _ in 0..n {
                        data.push(PaletteItem::color([i, j, k]));
                        k += ds;
                    }
                    j += ds;
                }
                i += ds;
            }    
        }

        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-system:palette:buffer"),
            usage: wgpu::BufferUsages::STORAGE,
            contents: bytemuck::cast_slice(&data),
        });

        Self { buffer, data }
    }
}


GPU! { struct Voxel { value: u32 } }
impl Voxel {
    fn new(id: u32) -> Self {
        let sdf = if id == 0 { 1 } else { 0 };

        Self {
            value: (id << 4) | sdf,
        }
    }
}


GPU! {
    struct Chunk {
        data: [[[Voxel; Chunk::RAD]; Chunk::RAD]; Chunk::RAD],
    }    
}
enum ChunkType {
    Air,
    Chunk(Chunk),
}

impl Chunk {
    const RAD: usize = 8;

    fn new() -> Self {
        let data = [[[Voxel::new(0); 8]; 8]; 8];
        Self { data }
    }
    fn _fill(&mut self, value: Voxel) {
        for slice in &mut self.data {
            for row in slice {
                for voxel in row {
                    *voxel = value;
                }
            }
        }
    }

    fn generate(perlin: &noise::Perlin, fbm: &noise::Fbm, palette: &Palette, level: usize, x: f64, y: f64, z: f64) -> ChunkType {
        let mut data = [[[Voxel::new(0); 8]; 8]; 8];

        let res = (1 << level) as f64;
        let s= 0.2 / (8.0 * res);
        let n = palette.data.len() as u32;

        let mut air = 0;

        fastrand::seed(0);

        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    let px = x * 8.0 + i as f64;
                    let py = y * 8.0 + j as f64;
                    let pz = z * 8.0 + k as f64;    

                    use noise::NoiseFn;
                    let surface = 96.0 + 32.0 * perlin.get([0.01 * px, 0.01 * pz]);

                    let id = if py < surface {                        
                        let val = fastrand::f32();
                        
                        if perlin.get([s * px, s * py, s * pz]) < 0.1 {
                            if val > 0.2 {
                                fastrand::u32(Palette::RANGE_GENERAL)
                            }
                            else if val > 0.03 {
                                fastrand::u32(Palette::RANGE_METAL)
                            }
                            else if val > 0.01 {
                                fastrand::u32(Palette::RANGE_LIGHT)
                            }
                            else {
                                0
                            }    
                        }
                        else {
                            0
                        }
                    } 
                    else { 0 };

                    if id == 0 {
                        air += 1;
                    }

                    data[i][j][k] = Voxel::new(id as _);                    
                }
            }
        }

        match air < 512 {
            true => ChunkType::Chunk(Chunk { data }),
            false => ChunkType::Air,
        }
    }
}



GPU! {
    struct ConfigData {
        offset: [[f32; 4]; MAX_LOD],
        pos: [[f32; 4]; MAX_LOD],
    }
}

struct Config {
    data: ConfigData,
    buffer: wgpu::Buffer
}
impl Config {
    fn new(device: &wgpu::Device) -> Self {
        let data = ConfigData {
            offset: [[0.0, 0.0, 0.0, 1.0]; MAX_LOD],
            pos: [[0.0, 0.0, 0.0, 1.0]; MAX_LOD],
        };
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-system:config:buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(&data),
        });

        Self { data, buffer }
    }
    fn update(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.data));
    }
}

#[derive(Clone, Copy)]
struct GridSize {
    shift_lod: u8,
    shift_r: u8,
    shift_x: u8,
    shift_y: u8,
    shift_z: u8,

    lod_levels: u8,
    r: u8,
    r_mask: usize,
    num_chunks: usize
}
GPU! {
    struct GridVector {
        lod: u8,
        x: u8,
        y: u8,
        z: u8,
    }    
}
type GridIndex = usize;

impl GridSize {
    fn new(lod_power: u8, r_power: u8) -> Self {
        let shift_r = r_power;
        let shift_z = 0;
        let shift_y = r_power;
        let shift_x = 2 * r_power;
        let shift_lod = 3 * r_power;

        let lod_levels = 1 << lod_power;
        let r = 1 << r_power;
        let r_mask = (r - 1) as _;

        let num_chunks = (1 as GridIndex) << (lod_power + 3 * r_power);

        Self {
            shift_r,
            shift_x,
            shift_y,
            shift_z,
            shift_lod,

            lod_levels,
            r,
            r_mask,
            num_chunks,
        }
    }
    fn pos_from_index(&self, index: GridIndex) -> GridVector {
        GridVector { 
            lod: (index >> self.shift_lod)as _, 
            x: ((index >> self.shift_x) & self.r_mask) as _, 
            y: ((index >> self.shift_y) & self.r_mask) as _,
            z: ((index >> self.shift_z) & self.r_mask) as _, 
        }
    }
    fn index_from_pos(&self, vec: &GridVector) -> GridIndex {
        (vec.lod as GridIndex) << self.shift_lod 
        | (vec.x as GridIndex) << self.shift_x 
        | (vec.y as GridIndex) << self.shift_y 
        | (vec.z as GridIndex)
    }
}


fn buffer_entry(
    binding: u32,
    binding_type: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: binding_type,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[derive(Debug)]
struct Partition {
    past_index: u32,
    index: u32,
    size: u32,
    offset: u32,
}
impl Partition {
    fn new(size: u32, offset: u32) -> Self {
        Self {
            past_index: 0,
            index: 0,
            size,
            offset
        }
    }

    fn next_index(&mut self) -> u32 {
        let old_index = self.index;
        self.index = (self.index + 1) % self.size;
        return self.offset + old_index;
    }
}
struct Cache {
    data: Vec<Chunk>,
    data_pos: Vec<Option<GridVector>>,

    buffer: wgpu::Buffer,
    partitions: Vec<Partition>, 
}
impl Cache {
    fn new(device: &wgpu::Device) -> Self {
        let mut offset = 0;
        let partitions = [
            0x80, 
            { let r = 32; 4 * r * r * r }
        ].iter().map(
            |&size| {
                let out = Partition::new(size, offset);
                offset += size;
                out
            }
        ).collect::<Vec<_>>();

        let len = offset as _;
        let data_pos = vec![None; len];
        let mut data = vec![Chunk::new(); len];

        data[0].data[0][0][0] = Voxel::new(1);

        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-system:system:buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&data),
        });

        Self { 
            data, 
            data_pos,

            buffer, 
            partitions
        }
    }

    fn push_chunk(&mut self, grid: &mut Grid, chunk: Chunk, pos: GridVector) -> u32 {    
        let index = self.partitions[1].next_index() as usize;

        if let Some(old_pos) = self.data_pos[index] {
            let old_index = grid.size.index_from_pos(&old_pos);
            grid.load[old_index] = false;
        } 

        self.data_pos[index] = Some(pos);
        self.data[index] = chunk;

        index as _
    }

    fn update(&mut self, queue: &wgpu::Queue) {
        fn write(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[Chunk], begin: usize, end: usize) {
            let data = bytemuck::cast_slice(&data[begin..end]);
            let offset = (begin * std::mem::size_of::<Chunk>()) as _;

            queue.write_buffer(buffer, offset, &data);
        }
        
        for part in self.partitions.iter_mut() {
            use std::cmp::Ordering::*;

            let buffer = &self.buffer;
            let data = &self.data[..];

            let begin = part.offset as _;
            let end = begin + part.size as usize;
            let i = begin + part.past_index as usize;
            let j = begin + part.index as usize;

            match part.past_index.cmp(&part.index) {
                Equal => {},
                Less => write(queue, buffer, data, i, j),
                Greater => {
                    write(queue, buffer, data, begin, j);
                    write(queue, buffer, data, i, end);
                }
            }

            part.past_index = part.index;
        }
    }
}

type ChunkID = u32;
GPU! { struct GridDesc { index: ChunkID } }
impl GridDesc {
    fn empty() -> Self {
        GridDesc { index: 0 }
    }
}
struct Grid {
    size: GridSize,
    data: Vec<GridDesc>,
    load: Vec<bool>,
    buffer: wgpu::Buffer,
}
impl Grid {
    fn new(device: &wgpu::Device) -> Self {
        let size = GridSize::new(0, 5);
        let data = vec![GridDesc::empty(); size.num_chunks];
        let load = vec![false; size.num_chunks];

        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-grid"),
            contents: &bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        });

        Grid { size, data, load, buffer }
    }

    fn load_sheet(&mut self, lod: usize, k: usize, swizzle: [usize; 3]) {
        let shift_r = self.size.shift_r as usize;
        
        let swizzle = [
            swizzle[0] * shift_r,
            swizzle[1] * shift_r,
            swizzle[2] * shift_r
        ];

        let lod = lod << self.size.shift_lod;
        let n = self.size.r as usize;
     
        for i in 0..n {
            for j in 0..n {
                let index = lod 
                    | i << swizzle[0] 
                    | j << swizzle[1] 
                    | k << swizzle[2];
                
                self.load[index] = false;
            }
        }
    }
    fn query_load(&mut self, lod: usize, disp: [i64; 3], offset: [i64; 3]) {
        let swizze_iter = [
            [0, 1, 2],
            [2, 0, 1],
            [1, 2, 0]
        ].iter().enumerate();

        for (n, &swizzle) in swizze_iter {
            let depth = disp[n].abs() as usize;
            let dir = disp[n].signum();

            if depth > 0 {
                for i in 0..depth {
                    let r = self.size.r as _;
                    let k = (offset[n] + (i as i64) * dir).rem_euclid(r) as _;

                    self.load_sheet(lod, k, swizzle);
                }
            }
        }
    }
    fn update(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.data));
    }
}

use std::{sync::{ Arc, mpsc::{ Receiver, SyncSender }}, ops::Range};

#[derive(Copy, Clone)]
struct WorkDesc {
    index: usize,
    level: usize,
    x: f64,
    y: f64,
    z: f64,
}

struct GenChunk {
    index: GridIndex,
    chunk: ChunkType
}
struct ChunkWorker {
    channel: SyncSender<WorkDesc>,
    _thread: std::thread::JoinHandle<()>,
}
impl ChunkWorker {
    fn new(sender: Arc<SyncSender<GenChunk>>, palette: Arc<Palette>) -> Self {
        let (channel, receiver) = std::sync::mpsc::sync_channel::<WorkDesc>(0x2000);

        let _thread = std::thread::spawn(move || {
            let perlin = noise::Perlin::new();
            let fbm = noise::Fbm::new();    

            while let Ok(work) = receiver.recv() {
                let level = work.level;
                let (x, y, z) = (work.x, work.y, work.z);

                let chunk = GenChunk {
                    index: work.index,
                    chunk: Chunk::generate(&perlin, &fbm, &palette, level, x, y, z),
                };

                if let Err(err) = sender.send(chunk) {
                    panic!("{}", err)
                }
            }
        });
        
        Self { channel, _thread }
    }
}
struct Generator {
    chunk_receiver: Receiver<GenChunk>,
    workers: Vec<ChunkWorker>,
    last_index: usize,
}
impl Generator {
    fn new(palette: Arc<Palette>) -> Self {
        let (chunk_sender, chunk_receiver) = std::sync::mpsc::sync_channel(0x1000);
        let chunk_sender = Arc::new(chunk_sender);

        let workers: Vec<_> = (0..4).map(
            |_| ChunkWorker::new(chunk_sender.clone(), palette.clone())
        ).collect();

        Self {
            chunk_receiver,
            workers,
            last_index: 0,
        }
    }
    fn push_work(&mut self, work: WorkDesc) -> Result<(), ()>{
        let len = self.workers.len();
        for _ in 0..len {
            self.last_index = (self.last_index + 1) % len;
            
            if let Ok(_) = self.workers[self.last_index].channel.try_send(work.clone()) {
                return Ok(());
            }
        }        
        Err(())
    }
}


pub struct World {
    origin: [[i64; 3]; MAX_LOD],
    generator: Generator,

    grid: Grid, 
    cache: Cache,
    config: Config,

    pub bindgroup: wgpu::BindGroup,
}
impl World {
    pub fn new(device: &wgpu::Device) -> Self {
        let palette = Palette::new(device);

        let grid = Grid::new(device);
        let cache = Cache::new(device);
        let config = Config::new(device);

        #[rustfmt::skip]
        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk-system:render:bindgroup"),
            layout: &Self::layout(device),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cache.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grid.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: palette.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: config.buffer.as_entire_binding() }        
            ],
        });

        let generator = Generator::new(Arc::new(palette));
        let origin = [[0; 3]; MAX_LOD];

        Self {
            origin,
            generator,

            grid,
            cache,
            config,

            bindgroup,
        }
    }
    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk-system:render:layout"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Uniform),
            ],
        })
    }

    fn generate(&mut self) {
        let mut count = 0; 

        let load = 0x800;

        let r = self.grid.size.r as i64;
        let grid_offset = (self.grid.size.r / 2) as i64;

        for (index, val) in self.grid.load.iter_mut().enumerate() {
            if !*val {
                let pos = self.grid.size.pos_from_index(index);

                let lod = pos.lod as usize;
                let off: [i64; 3] = self.origin[lod];

                let ox = off[0];
                let oy = off[1];
                let oz = off[2];
 
                let work = WorkDesc {
                    index,
                    level: pos.lod as _,
                    x: ((pos.x as i64 - ox).rem_euclid(r) + ox) as f64,
                    y: ((pos.y as i64 - oy).rem_euclid(r) + oy) as f64,
                    z: ((pos.z as i64 - oz).rem_euclid(r) + oz) as f64,
                };

                if let Ok(_) = self.generator.push_work(work) {
                    *val = true;
                }

                count += 1;
                if count > load {
                    break;
                }
            }
        }

        let mut loaded = 0;
        for _ in 0..load {
            if let Ok(GenChunk { index, chunk }) = self.generator.chunk_receiver.try_recv() {
                match chunk {
                    ChunkType::Air => {
                        self.grid.data[index].index = 1;
                    },
                    ChunkType::Chunk(chunk) => {
                        let pos = self.grid.size.pos_from_index(index);
                        self.grid.data[index].index = self.cache.push_chunk(&mut self.grid, chunk, pos);
                    }
                }    
                loaded += 1;
            }
            else {
                break;
            }
        }

        
        println!("=== Cache ===");
        println!("Loaded: {}", loaded);

        for (i, part) in self.cache.partitions.iter().enumerate() {
            let len = part.size;
            let begin = part.offset;
            let end = begin + len;

            let dim = (begin, end);
            let slice = (part.past_index, part.index);
            println!("P[{}]: {:?}[{}] : {:?}", i, dim, len, slice);
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, new_origin: [f64; 3]) {
        {
            const CHUNK_RAD: f64 = Chunk::RAD as _;
            let grid_rad = 0.5 * CHUNK_RAD * self.grid.size.r as f64;

            let mut res = 1.0;
        
            for lod in 0..self.grid.size.lod_levels as _ {
                let inv_rad = 1.0 / (CHUNK_RAD * res);
                let mut disp = [0; 3];

                let w = res * CHUNK_RAD;
                let mut new_offset = [0; 3];

                for n in 0..3 {
                    disp[n] = (new_origin[n] * inv_rad - self.origin[lod][n] as f64).floor() as _;

                    let offset = (new_origin[n] / w).floor();
                    let chunk_pos = CHUNK_RAD * offset;

                    self.config.data.offset[lod][n] = (chunk_pos - grid_rad) as _;
                    self.config.data.pos[lod][n] = (new_origin[n] / res - chunk_pos) as _;

                    new_offset[n] = offset as _; 
                }

                self.grid.query_load(lod, disp, self.origin[lod]);

                for n in 0..3 {
                    self.origin[lod][n] += disp[n];
                }
            

                res *= 2.0;
            }
        }

        self.generate();
        self.cache.update(queue);
        self.grid.update(queue);
        self.config.update(queue);
    }
}