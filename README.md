# Voxel Renderer

Written in Rust + [wgpu](https://github.com/gfx-rs/wgpu)

# Demo

![](https://github.com/ankrisac/Voxel-Renderer/blob/main/demo/Demo-1.gif)
- Raytraced shadows, procedural generation

![](https://github.com/ankrisac/Voxel-Renderer/blob/main/demo/Demo-2.gif)
- Metallic reflection

# Architecture
```mermaid
flowchart LR
  PChunk[Chunk System]

  subgraph PWorld[WorldRender]
    direction LR
    Camera --> voxel

    voxel --if-metal--> voxel'
    voxel --> sun
        
    voxel' --> sun
  end

  subgraph PEffect[PostProcess]
    direction LR
    output --upscale--> Target
    Target --color correct--> Surface
  end
    
  PChunk --> PWorld
  PWorld --> PEffect
  PEffect --> Surface
```

The raytracer uses a modified version of the algorithm presented in [A Fast Voxel Traversal Algorithm (1987) - Amanatides & Woo](http://www.cse.yorku.ca/~amana/research/grid.pdf)
