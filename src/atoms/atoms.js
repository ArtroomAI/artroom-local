import { atom } from 'recoil';

export const serverStatusState = atom({
    key: 'serverStatus',
    default: ''
});

export const CFGScaleState = atom({
    key: 'cfg_scale',
    default: ''
});

export const aspectRatioState = atom({
    key: 'aspect_ratio',
    default: ''
});

export const aspectRatioSelectionState = atom({
    key: 'aspectRatioSelection',
    default: ''
});

export const batchNameState = atom({
    key: 'batch_name',
    default: ''
});

export const heightState = atom({
    key: 'height',
    default: ''
});

export const widthState = atom({
    key: 'width',
    default: ''
});


export const imageSavePathState = atom({
    key: 'image_save_path',
    default: ''
});

export const longSavePathState = atom({
    key: 'long_save_path',
    default: false
});

export const highresFixState = atom({
    key: 'highres_fix',
    default: false
});

export const initImageState = atom({
    key: 'init_image',
    default: ''
});

export const initImagePathState = atom({
    key: 'initImagePath',
    default: ''
});

export const keepWarmState = atom({
    key: 'keep_warm',
    default: ''
});

export const ckptState = atom({
    key: 'ckpt',
    default: ''
});

export const ckptDirState = atom({
    key: 'ckpt_dir',
    default: ''
});

export const nIterState = atom({
    key: 'n_iter',
    default: ''
});

export const navSizeState = atom({
    key: 'navSize',
    default: 'small'
});

export const negativePromptsState = atom({
    key: 'negative_prompts',
    default: ''
});

export const textPromptsState = atom({
    key: 'text_prompts',
    default: ''
});

export const neonVRAMState = atom({
    key: 'neon_vram',
    default: false
});

export const openPicturesState = atom({
    key: 'open_pictures',
    default: ''
});

export const paintTypeState = atom({
    key: 'paint_type',
    default: 'Use Mask'
});

export const queueRunningState = atom({
    key: 'queueRunning',
    default: false
});


export const queueState = atom({
    key: 'queue',
    default: []
});


export const runningState = atom({
    key: 'running',
    default: false
});


export const samplerState = atom({
    key: 'sampler',
    default: ''
});


export const saveGridState = atom({
    key: 'save_grid',
    default: false
});

export const debugMode = atom({
    key: 'debug_mode',
    default: false
});


export const seedState = atom({
    key: 'seed',
    default: ''
});

export const speedState = atom({
    key: 'speed',
    default: ''
});

export const stepsState = atom({
    key: 'steps',
    default: ''
});

export const strengthState = atom({
    key: 'strength',
    default: ''
});

export const useCPUState = atom({
    key: 'use_cpu',
    default: false
});

export const useRandomSeedState = atom({
    key: 'use_random_seed',
    default: false
});

export const initImageThumbnailState = atom({
    key: 'initImageThumbnail',
    default: { 'ImagePath': '',
        'B64': '' }
});

export const mainImageState = atom({
    key: 'mainImage',
    default: ''
});

export const latestImageState = atom({
    key: 'latestImage',
    default: []
});

export const latestImagesIDState = atom({
    key: 'latestImagesID',
    default: 0
});

export const imageViewPathState = atom({
    key: 'imageViewPath',
    default: ''
});

export const delayState = atom({
    key: 'delay',
    default: 1
});

export const paintHistoryState = atom({
    key: 'paintHistory',
    default: []
});

export const cloudModeState = atom({
    key: 'cloudMode',
    default: false
});


