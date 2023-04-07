import { atom } from 'recoil';

export const imageModalB64State = atom({
    key: 'imageModalB64',
    default: ''
});

export const imageModalMetadataState = atom<ImageMetadata>({
    key: 'imageModalMetadata',
    default: {
        text_prompts: '',
        negative_prompts: '',
        W: '',
        H: '',
        seed: '',
        sampler: '',
        steps: '',
        strength: '',
        cfg_scale: '',
        ckpt: '',
        vae: ''
    }
});

export const showImageModalState = atom({
    key: 'showImageModal',
    default: false
});

export const showLoginModalState = atom({
    key: 'showLoginModal',
    default: false
});
