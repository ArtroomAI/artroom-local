interface Lora {
    name: string;
    weight: number; // float
}

interface QueueType {
    text_prompts: string;
    negative_prompts: string;

    // image to image options
    strength: number;
    init_image: string;

	width: number;
	height: number;
    
    // models
    models_dir: string;
    ckpt: string; // absolute path
    vae: string; // absolute path
    lora: Lora[];
    controlnet: string;
    use_preprocessed_controlnet: boolean; 
    remove_background: string;
    use_removed_background: boolean; 

    // sampler options
    sampler: string;
    steps: number;
	cfg_scale: number;
    clip_skip: number;
    seed: number;

    // inpainting options
    mask_image: string;
    invert: boolean;
    palette_fix: boolean;

    //
    image_save_path: string; // absolute path

    // generation options
    show_intermediates: boolean;
    n_iter: number;
    save_grid: boolean;
    speed: string;
    device?: string; // ? CPU / GPU
    long_save_path: boolean;
    highres_fix: boolean;
    id: string;
}

interface QueueTypeWithIndex extends QueueType {
    key: number;
    index: number;
    lastItem: boolean;
}