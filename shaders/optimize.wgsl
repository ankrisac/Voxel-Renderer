let ZERO: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
let UNIT: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);


struct Chunk {
    data: array<array<array<u32, 8>, 8>, 8>;
    lights: array<vec4<f32>, 16>;
};
[[block]] struct ChunkSystem {
    data: array<array<array<Chunk, 32>, 32>, 32>;
};

[[block]] struct WorkQueue {
    data: array<vec4<u32>>;
};

[[group(0), binding(0)]] var<storage, read_write> chunk_system: ChunkSystem;
[[group(0), binding(1)]] var<storage, read> work_sdf: WorkQueue;

fn voxelSDF(level: i32, pos: vec3<i32>) -> u32 {  
    let P = vec3<u32>(pos % 256); 
    let ps = P >> vec3<u32>(3u, 3u, 3u);
    let pc = P & vec3<u32>(7u, 7u, 7u);

    return chunk_system.data[ps.x][ps.y][ps.z].data[pc.x][pc.y][pc.z] & 15u;    
}

let I: vec3<i32> = vec3<i32>(1, 0, 0);
let J: vec3<i32> = vec3<i32>(0, 1, 0);
let K: vec3<i32> = vec3<i32>(0, 0, 1);

fn LOD_kernel(level: i32, pos: vec3<u32>) {
    let ps = pos >> vec3<u32>(3u, 3u, 3u);
    let pc = pos & vec3<u32>(7u, 7u, 7u);

    let voxel = chunk_system.data[ps.x][ps.y][ps.z].data[pc.x][pc.y][pc.z];
    let center = voxel & 15u;

    if(center > 0u) {
        let P = vec3<i32>(pos);
    
        let s1 = min(voxelSDF(level, P - I), voxelSDF(level, P + I));
        let s2 = min(voxelSDF(level, P - J), voxelSDF(level, P + J));
        let s3 = min(voxelSDF(level, P - K), voxelSDF(level, P + K));

        let sdf = min(min(s1, min(s2, s3)) + 1u, 15u);

        //TODO: Use hex constants instead when Naga fixes the parsing bug
        let id = voxel >> 4u;
        let voxel = (id << 4u) | sdf; 
        
        chunk_system.data[ps.x][ps.y][ps.z].data[pc.x][pc.y][pc.z] = voxel;
    }
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn main([[builtin(global_invocation_id)]] pos: vec3<u32>) {
    LOD_kernel(0, pos);
}