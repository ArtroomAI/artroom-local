function randomIntFromInterval(min: number, max: number) {
  // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min)
}

function isKeyOf<T extends object>(key: string | number | symbol, object: T): key is keyof T {
  return key in object
}

export function parseSettings(settings: QueueType, useRandom: boolean) {
  settings.seed = useRandom ? randomIntFromInterval(1, 4294967295) : settings.seed

  const sampler_format_mapping = {
    k_euler: 'euler',
    k_euler_ancestral: 'euler_a',
    k_dpm_2: 'dpm',
    k_dpm_2_ancestral: 'dpm_a',
    k_lms: 'lms',
    k_heun: 'heun',
  }
  if (isKeyOf(settings.sampler, sampler_format_mapping)) {
    settings.sampler = sampler_format_mapping[settings.sampler]
  }

  return settings
}
