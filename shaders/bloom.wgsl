struct Invocation {
    [[builtin(workgroup_id)]] work_id: vec3<u32>;

    [[builtin(global_invocation_id)]] global_id: vec3<u32>;
    [[builtin(local_invocation_id)]] local_id: vec3<u32>;    
    [[builtin(local_invocation_index)]] local_index: u32;
};

[[group(0), binding(0)]] var texture: texture_storage_2d<rgba32float, read_write>;

let ZERO: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
let UNIT: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

fn bloom_color(uv: vec2<i32>) -> vec3<f32> {
    let color = textureLoad(texture, uv).xyz;
    return max(color - UNIT, ZERO);
}

fn color_correct(color: vec3<f32>) -> vec3<f32> {
    let luma = dot(color, vec3<f32>(0.299, 0.587, 0.114));

    var sat: f32 = 1.0;
    var color: vec3<f32> = color;
    for (var i: i32 = 0; i < 3; i = i + 1){
        let pix = color[i];
        let inv = 1.0 / (luma - pix);

        if (pix > 1.0) {
            sat = min(sat, (luma - 1.0) * inv);      
        }
        if (pix < 0.0) {
            sat = min(sat, luma * inv);
        }
    }
    sat = clamp(sat, 0.0, 1.0);

    for (var i: i32 = 0; i < 3; i = i + 1) {
        color[i] = clamp((color[i] - luma) * sat + luma, 0.0, 1.0);
    }

    return color;
} 

[[stage(compute), workgroup_size(8, 8, 8)]]
fn main(input: Invocation) {
    let uv = vec2<i32>(input.global_id.xy);
    let input_color = textureLoad(texture, uv);
    let color = input_color.rgb;
    let step = input_color.a;

    let color = vec4<f32>(color_correct(color), step + 1.0);

    textureStore(texture, uv, color);
}