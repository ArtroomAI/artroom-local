declare global {
  interface Lora {
    name: string
    weight: number
    /**
     * @deprecated use name instead
     */
    path?: string
  }
  interface ExifDataType {
    /**
     * @type int
     * @deprecated use width instead
     */
    W?: number
    /**
     * @type int
     * @deprecated use height instead
     */
    H?: number
    /**
     * @type int
     */
    width: number
    /**
     * @type int
     */
    height: number
    cfg_scale: number
    ckpt: string
    controlnet: null | string
    loras: Lora[]
    negative_prompts: string
    sampler: string
    /**
     * @type int
     */
    seed: number
    /**
     * @type int
     */
    steps: number
    strength: number
    text_prompts: string
    vae: string
    /**
     * @type int
     */
    clip_skip: number
  }
}

export const DEFAULT_EXIF: ExifDataType = {
  width: 0,
  height: 0,
  cfg_scale: 0,
  ckpt: '',
  controlnet: 'none',
  loras: [],
  negative_prompts: '',
  sampler: '',
  seed: 0,
  steps: 0,
  strength: 0,
  text_prompts: '',
  vae: '',
  clip_skip: 1,
}
