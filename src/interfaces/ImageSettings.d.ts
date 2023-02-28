interface QueueType {
    text_prompts: string;
    negative_prompts: string;

    // image to image options
    strength: number;
    init_image: string;

	width: number;
	height: number;
    
    // models
    ckpt: string; // absolute path
    vae: string; // absolute path
    controlnet: string;

    // sampler options
    sampler: string;
    steps: number;
	cfg_scale: number;
    seed: number;

    // inpainting options
    mask_image: string;
    invert: boolean;
    palette_fix: boolean;

    //
    image_save_path: string; // absolute path

    // generation options
    n_iter: number;
    save_grid: boolean;
    speed: string;
}
