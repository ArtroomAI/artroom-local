import { atom } from 'recoil';
import { ImageSettings, AppSettings } from './atoms.types';

export const imageSettingsState = atom<ImageSettings>({
    key: 'imageSettings',
    default: {
        text_prompts: '',
        negative_prompts: '',
        batch_name: '',
        n_iter: 3,
        steps: 50,
        strength: 0.75,
        cfg_scale: 12,
        sampler: '',
        width: 512,
        height: 512,
        aspect_ratio : '1:1',
        ckpt: '',
        vae: '',
        seed: 5,
        speed: 'High',
        save_grid: false,
        use_random_seed: false,
        init_image: '',
        mask_image: '',
        invert: false,
        image_save_path: '',
        ckpt_dir: '',
    }
})

export const appSettingsState = atom<AppSettings>({
    key: 'appSettings',
    default: {
        text_prompts: '',
        negative_prompts: '',
        batch_name: '',
        n_iter: 3,
        steps: 50,
        strength: 0.75,
        cfg_scale: 12,
        sampler: '',
        width: 512,
        height: 512,
        ckpt: '',
        vae: '',
        seed: 5,
        save_grid: false,
        use_random_seed: false,
        init_image: '',
        mask_image: '',
    }
})

export const serverStatusState = atom({
    key: 'serverStatus',
    default: ''
});

export const aspectRatioSelectionState = atom({
    key: 'aspectRatioSelection',
    default: 'None'
});

export const longSavePathState = atom({
    key: 'long_save_path',
    default: false
});

export const highresFixState = atom({
    key: 'highres_fix',
    default: false
});

export const initImagePathState = atom({
    key: 'initImagePath',
    default: ''
});

export const ckptsState = atom({
    key: 'ckpts',
    default: []
});

export const navSizeState = atom({
    key: 'navSize',
    default: 'small'
});

export const queueRunningState = atom({
    key: 'queueRunning',
    default: false
});


export const queueState = atom<QueueType[]>({
    key: 'queue',
    default: []
});


export const runningState = atom({
    key: 'running',
    default: false
});

export const cloudRunningState = atom({
    key: 'cloud_running',
    default: false
});

export const debugMode = atom({
    key: 'debug_mode',
    default: false
});

export const initImageThumbnailState = atom({
    key: 'initImageThumbnail',
    default: 
    { 'ImagePath': '',
        'b64': '' }
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

export * from './atoms.modals'
export * from './atoms.login'
