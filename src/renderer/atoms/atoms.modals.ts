import { atom } from 'recoil'

export const imageModalB64State = atom({
  key: 'imageModalB64',
  default: '',
})

export const imageModalMetadataState = atom<ExifDataType>({
  key: 'imageModalMetadata',
  default: {
    text_prompts: '',
    negative_prompts: '',
    width: 0,
    height: 0,
    seed: 0,
    sampler: '',
    steps: 0,
    strength: 0,
    cfg_scale: 0,
    ckpt: '',
    vae: '',
    loras: [],
    controlnet: 'none',
    clip_skip: 0,
  },
})

export const showImageModalState = atom({
  key: 'showImageModal',
  default: false,
})

export const showLoginModalState = atom({
  key: 'showLoginModal',
  default: false,
})
