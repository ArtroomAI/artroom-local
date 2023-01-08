interface QueueType {
    height: number;
    width: number;
    cfg_scale: string;
    ckpt: string;
    device: string;
    id: string;
    init_image: string;
    invert: string;
    keep_warm: string;
    key: number;
    mask: string;
    n_iter: string;
    negative_prompts: string;
    text_prompts: string;
    sampler: string;
    seed: string;
    skip_grid: string;
    steps: string;
    strength: string;
}

interface QueueTypeWithIndex extends QueueType {
    index: number;
    lastItem: boolean;
}