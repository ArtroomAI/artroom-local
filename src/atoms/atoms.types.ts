export type ImageSettings = {
    text_prompts: string;
    negative_prompts: string;
    batch_name: string;
    n_iter: number;
    steps: number;
    strength: number;
	cfg_scale: number;
    sampler: string;
	width: number;
	height: number;
    aspect_ratio: string;
    ckpt: string;
    vae: string;
    seed: number;
    speed: string;
    save_grid: boolean;
    use_random_seed: boolean;
    init_image: string;
    mask_image: string;
    invert: boolean;
    image_save_path: string;
    ckpt_dir: string;
    palette_fix: boolean;
};

export type AppSettings = {
    text_prompts: string;
    negative_prompts: string;
    batch_name: string;
    n_iter: number;
    steps: number;
    strength: number;
	cfg_scale: number;
    sampler: string;
	width: number;
	height: number;
    ckpt: string;
    vae: string;
    seed: number;
    save_grid: boolean;
    use_random_seed: boolean;
    init_image: string;
    mask_image: string;
};

export interface ImageState {
    b64: string;
    path: string;
    batch_id: number
}
