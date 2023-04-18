interface ExifDataType {
    width: number;
    height: number;
    cfg_scale: number;
    ckpt: string;
    controlnet: null | string;
    loras: Lora[];
    negative_prompts: string;
    sampler: string;
    seed: number;
    steps: number;
    strength: number;
    text_prompts: string;
    vae: string;
    clip_skip: number;
}
