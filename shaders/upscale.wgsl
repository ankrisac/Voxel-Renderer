struct Vertex {
    [[builtin(vertex_index)]] index: u32;
    [[location(0)]] pos: vec3<f32>;
};
struct Fragment {
    [[builtin(position)]] pos: vec4<f32>;
};

[[stage(vertex)]] 
fn main(input: Vertex) -> Fragment {
    var output: Fragment;
    output.pos = vec4<f32>(input.pos, 1.0);
    return output;
}

[[block]] 
struct UpscaleData {
    origin: vec2<f32>;
    inv_height: f32;
    scale: f32;
};

[[group(0), binding(0)]] var input_texture: texture_storage_2d<rgba32float, read>;
[[group(0), binding(1)]] var<uniform> upscale_config: UpscaleData;

[[stage(fragment)]] 
fn main(input: Fragment) -> [[location(0)]] vec4<f32> {
    let uv = vec2<i32>(input.pos.xy / upscale_config.scale);
    let color = textureLoad(input_texture, uv).xyz;

    return vec4<f32>(color, 1.0);
}