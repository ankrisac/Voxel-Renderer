let ZERO: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
let HALF: vec3<f32> = vec3<f32>(0.5, 0.5, 0.5);
let UNIT: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

[[block]] 
struct Camera { 
    transform: mat4x4<f32>; 
};

[[group(1), binding(0)]] 
var<uniform> camera: Camera;

type VoxelID = u32;
type ChunkID = u32;

let DIM_CHUNK: u32 = 8u;
let DIM_BLOCK: u32 = 32u; 

let DIM_GRID: f32 = 256.0; 
let RAD_GRID: f32 = 128.0; 

struct Chunk { 
    data: array<array<array<VoxelID, DIM_CHUNK>, DIM_CHUNK>, DIM_CHUNK>; 
};
struct GridDesc { 
    index: ChunkID; 
};


[[block]] struct ChunkGrid { 
    data: array<array<array<GridDesc, DIM_BLOCK>, DIM_BLOCK>, DIM_BLOCK>; 
};
[[block]] struct ChunkCache { data: array<Chunk, 128000>; };

struct VoxelType {
    color: vec3<f32>;
    lum: f32;
};
[[block]] struct ChunkPalette { 
    data: array<VoxelType, 32768>; 
};

[[block]] struct ChunkConfig {
    offset: array<vec4<f32>, 8>;
    pos: array<vec4<f32>, 8>;
};

[[group(2), binding(0)]] var<storage, read> chunk_cache: ChunkCache;
[[group(2), binding(1)]] var<storage, read_write> chunk_grid: ChunkGrid;
[[group(2), binding(2)]] var<storage, read> chunk_palette: ChunkPalette;
[[group(2), binding(3)]] var<uniform> chunk_config: ChunkConfig;

fn getVoxelAbsolute(level: i32, pos: vec3<u32>) -> VoxelID {
    let ps = pos >> vec3<u32>(3u, 3u, 3u);
    let pc = pos & vec3<u32>(7u, 7u, 7u);

    let index = chunk_grid.data[ps.x][ps.y][ps.z].index;    
    return chunk_cache.data[index].data[pc.x][pc.y][pc.z];
}

fn getVoxel(level: i32, pos: vec3<f32>) -> VoxelID {    
    let P = floor(pos + chunk_config.offset[level].xyz) % DIM_GRID;
    return getVoxelAbsolute(level, vec3<u32>(P));
}
fn getVoxelType(voxel: VoxelID) -> VoxelType {
    return chunk_palette.data[u32(voxel) >> 4u];
}

fn voxelSDF(voxel: VoxelID) -> u32 { return u32(voxel) & 15u; }

struct Intersection {
    pos: vec3<f32>;
    dist: f32;

    pos_vox: vec3<f32>;
    voxel: VoxelID;
    
    collided: bool;
};

fn raycast(max_steps: i32, max_dist: f32, level: i32, origin: vec3<f32>, dR: vec3<f32>) -> Intersection {
    let stepDir = sign(dR);
    let stepDirNegative = dR <= ZERO;

    let R1 = dR.yzx / dR.xyz;
    let R2 = R1 * R1;
    let dL = sqrt(UNIT + R2.xyz + UNIT / R2.zxy);

    var voxCoord: vec3<f32> = floor(origin);
    let microCoord = origin - voxCoord;

    let aligned = microCoord <= UNIT * 0.001;

    voxCoord = voxCoord - select(ZERO, UNIT, aligned && stepDirNegative);
    var L: vec3<f32> = dL * select(select(UNIT - microCoord, microCoord, stepDirNegative), UNIT, aligned);
    var min_L: f32 = min(L.x, min(L.y, L.z));

    var out: Intersection;
    out.collided = false;

    let voxel = getVoxel(level, voxCoord);    
    if(voxelSDF(voxel) == 0u) {
        out.voxel = voxel;
        out.collided = true;
        return out;
    }

    let max_dist_sq = 0.8 * max_dist * max_dist;
    var skip_dist: u32 = 0u;
    for(var i: i32 = 0; i < max_steps; i = i + 1) {
        let N = min(skip_dist, 16u);
        for(var j: u32 = 1u; j < N && min_L < max_dist; j = j + 1u) {
            let skipMag = step(L, UNIT * min_L);
            L = L + dL * skipMag;
            min_L = min(L.x, min(L.y, L.z));
            voxCoord = voxCoord + stepDir * skipMag;
        }

        let stepMag = step(L, UNIT * min_L);
        voxCoord = voxCoord + stepDir * stepMag;

        if (dot(voxCoord, voxCoord) > max_dist_sq) {
            break;
        }

        let voxel = getVoxel(level, voxCoord); 
        skip_dist = voxelSDF(voxel);

        if(skip_dist == 0u) {
            out.pos_vox = voxCoord;
            out.voxel = voxel;
            out.collided = true;
            break;
        }
        
        L = L + dL * stepMag;
        min_L = min(L.x, min(L.y, L.z)); 
    }

    out.pos = origin + dR * min_L;
    out.dist = min_L;
    return out;
}



fn getNormal(micro_pos: vec3<f32>, dR: vec3<f32>) -> vec3<f32> {
    let aligned = abs(micro_pos - HALF) >= 0.4995 * UNIT;
    return select(ZERO, -sign(dR), aligned);
}
fn getLum_ambient(normal: vec3<f32>) -> f32 {
    let dir = (normal + UNIT) * 0.5;
    let dir_weight = vec3<f32>(0.0, 1.0, 0.5);
    return 0.4 + 0.6 * dot(dir, dir_weight);
}
fn getLum_sun(level: i32, surface: Intersection, normal: vec3<f32>) -> vec3<f32> {
    let ray = normalize(vec3<f32>(1.0, 10.0, 1.5));
    let diff = dot(normal, ray);
    let dark = 0.001 * UNIT;

    if (diff > 0.0) {
        let max_dist = RAD_GRID;

        let specular = max(dot(normal, ray), 0.0);
        let shadow_int = raycast(256, max_dist, level, surface.pos, ray);

        let in_shadow = shadow_int.collided;
        let bright = diff * UNIT;
    
        return select(bright, dark, in_shadow);
    }

    return dark;
}


fn SDF_box(p: vec3<f32>) -> f32 {
    return length(max(abs(p + HALF) - HALF, ZERO));
}
fn getLum_light(level: i32, surface: Intersection, normal: vec3<f32>) -> vec3<f32> {    
    let V = floor(surface.pos_vox + chunk_config.offset[level].xyz) % DIM_GRID;
    let P = (surface.pos + chunk_config.offset[level].xyz) % DIM_GRID;
    let ps = vec3<u32>(V) >> vec3<u32>(3u, 3u, 3u);

    var output: vec3<f32> = ZERO;

    for(var i: i32 = 0; i < 16; i = i + 1) {
        let lum = vec4<f32>(0.0, 0.0, 0.0, 0.0); //chunk_system.data[ps.x][ps.y][ps.z].lights[i];

        if (lum.a > 0.0) {
            let voxel = getVoxelAbsolute(level, vec3<u32>(lum.xyz));
            let desc = getVoxelType(voxel);
            let color = desc.color;

            let ray = lum.xyz - P;
            let dist = SDF_box(ray);
            let ray = normalize(ray);

            if (dot(normal, ray) > 0.0) {
                let collision = raycast(16, dist + 1.0, level, surface.pos, ray);
                
                if (collision.dist > dist) {
                    output = output + color.xyz * desc.lum * pow(4.0 * dist, -2.0);
                }
            }
        }
    }

    return output;
}

fn SDF_f32(level: i32, p: vec3<f32>) -> f32 {
    let voxel = getVoxel(level, p);
    let sdf = voxelSDF(voxel);

    return min(f32(sdf), 1.0);
}
fn AO_corner(e1: f32, v: f32, e2: f32) -> f32 {
    let ao_level = select(e1 + v + e2, 0.0, e1 + e2 < 0.1);
    return (1.0 + ao_level) * 0.25;
}
fn getLum_occlusion(level: i32, surface: Intersection, normal: vec3<f32>) -> vec3<f32> {
    let pos = surface.pos + 0.0001 * normal;
    let vox = surface.pos_vox + normal;

    var uv: vec3<f32> = pos - vox;

    var I: vec3<f32> = ZERO;
    var J: vec3<f32> = ZERO;

    var i: i32 = 0;
    var j: i32 = 0;

    let N = abs(normal);
    if (N.x > 0.5) {
        uv.x = 0.0;
        i = 2;
        j = 1;
    }
    else {
        if (N.z > 0.5) {
            uv.z = 0.0;
            i = 0;
            j = 1;
        }
        else {
            uv.y = 0.0;
            i = 0;
            j = 2;
        }
    }    
    I[i] = 1.0;
    J[j] = 1.0;

    //02 12 22
    //01 11 21
    //00 10 20

    let v00 = SDF_f32(level, vox - I - J);
    let v20 = SDF_f32(level, vox + I - J);
    let v02 = SDF_f32(level, vox - I + J);
    let v22 = SDF_f32(level, vox + I + J);

    let v10 = SDF_f32(level, vox - J);
    let v12 = SDF_f32(level, vox + J);
    let v01 = SDF_f32(level, vox - I);
    let v21 = SDF_f32(level, vox + I);

    //01 11
    //00 10

    let q00 = AO_corner(v01, v00, v10);
    let q10 = AO_corner(v10, v20, v21);
    let q01 = AO_corner(v01, v02, v12);
    let q11 = AO_corner(v12, v22, v21); 

    let taxi_dst = dot(abs(uv), UNIT);

    var ao: f32 = 0.0;
    if (taxi_dst < 1.0) {
        let L10 = uv[i];
        let L01 = uv[j];
        let L00 = 1.0 - L10 - L01;
        
        ao = q00 * L00 + q10 * L10 + q01 * L01;
    }
    else {
        let L01 = 1.0 - uv[i];
        let L10 = 1.0 - uv[j];
        let L11 = 1.0 - L10 - L01;

        ao = q10 * L10 + q01 * L01 + q11 * L11;
    }

    return UNIT * ao;
}

[[group(0), binding(0)]] var scene_texture: texture_storage_2d<rgba32float, read_write>;

fn getLum(level: i32, S: Intersection, N: vec3<f32>) -> vec3<f32> {
    let I = vec3<f32>(1.0, 0.0, 0.0);
    let J = vec3<f32>(0.0, 1.0, 0.0);
    let K = vec3<f32>(0.0, 0.0, 1.0);

    let sun = getLum_sun(0, S, N);
    let light = getLum_light(0, S, N);

    return 0.01 + light + sun; 
}
fn blitRay(dR: vec3<f32>, uv: vec2<i32>) {
    var S: Intersection = raycast(512, RAD_GRID, 0, chunk_config.pos[0].xyz, dR);

    let epsilon = 0.0001;
    let K = 1.0;

    if(S.collided) {
        let voxel = getVoxelType(S.voxel);

        var color: vec3<f32> = voxel.color;
        if(voxel.lum == 0.0) {
            let N = getNormal(S.pos - S.pos_vox, dR);
            let ao = getLum_occlusion(0, S, N);  
            
            S.pos = S.pos + N * epsilon;

            color = color * ao * getLum(0, S, N);
        }
        else {
            if(voxel.lum > 0.0) {
                color = color * voxel.lum;
            }
            else {
                let N = getNormal(S.pos - S.pos_vox, dR);

                let ao = getLum_occlusion(0, S, N);  
                S.pos = S.pos + N * epsilon;

                let rdR = reflect(dR, N);
                var rS: Intersection = raycast(32, 64.0, 0, S.pos, rdR);

                let rvoxel = getVoxelType(rS.voxel);
                let spec = max(-voxel.lum, 1.0);

                var rcolor: vec3<f32> = rvoxel.color;
                if (rvoxel.lum > 0.0) {
                    rcolor = rcolor * rvoxel.lum;
                }
                else {
                    let rN = getNormal(rS.pos - rS.pos_vox, rdR);
                    let rao = getLum_occlusion(0, rS, rN);
                    rS.pos = rS.pos + rN * epsilon;
                    
                    rcolor = rcolor * getLum(0, rS, rN) * rao;
                }

                color = color * (1.0 - spec) * getLum(0, S, N) * ao + spec * rcolor;
            }
        }
        
        textureStore(scene_texture, uv, vec4<f32>(color, 0.0));
        return;
    }

    textureStore(scene_texture, uv, vec4<f32>(0.1, 0.2, 1.0, 0.0));
}

struct Invocation {
    [[builtin(global_invocation_id)]] global_id: vec3<u32>;
    [[builtin(local_invocation_id)]] local_id: vec3<u32>;    
    [[builtin(local_invocation_index)]] local_index: u32;
};

[[block]] struct UpscaleData {
    origin: vec2<f32>;
    inv_height: f32;
    scale: f32;
};
[[group(0), binding(1)]] var<uniform> upscale_config: UpscaleData;


[[stage(compute), workgroup_size(32, 32)]]
fn main(input: Invocation) {
    let uv = vec2<i32>(input.global_id.xy);
    let pos = vec2<f32>(input.global_id.xy);

    let coords = (pos * upscale_config.inv_height - upscale_config.origin);
    let dR = normalize((camera.transform * vec4<f32>(coords.x, -coords.y, 1.0, 1.0)).xyz);

    blitRay(dR, uv);
}