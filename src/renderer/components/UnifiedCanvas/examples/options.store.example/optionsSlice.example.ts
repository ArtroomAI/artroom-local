// import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Image } from 'painter'
import { FACETOOL_TYPES } from 'constants'

export type UpscalingLevel = 2 | 4

export type FacetoolType = typeof FACETOOL_TYPES[number]

export interface OptionsState {
  activeTab: number
  cfgScale: number
  codeformerFidelity: number
  currentTheme: string
  facetoolStrength: number
  facetoolType: FacetoolType
  height: number
  hiresFix: boolean
  img2imgStrength: number
  infillMethod: string
  initialImage?: Image | string // can be an Image or url
  isLightBoxOpen: boolean
  iterations: number
  maskPath: string
  optionsPanelScrollPosition: number
  perlin: number
  prompt: string
  sampler: string
  seamBlur: number
  seamless: boolean
  seamSize: number
  seamSteps: number
  seamStrength: number
  seed: number
  seedWeights: string
  shouldFitToWidthHeight: boolean
  shouldGenerateVariations: boolean
  shouldHoldOptionsPanelOpen: boolean
  shouldLoopback: boolean
  shouldPinOptionsPanel: boolean
  shouldRandomizeSeed: boolean
  shouldRunESRGAN: boolean
  shouldRunFacetool: boolean
  shouldShowOptionsPanel: boolean
  showDualDisplay: boolean
  steps: number
  threshold: number
  tileSize: number
  upscalingLevel: UpscalingLevel
  upscalingStrength: number
  variationAmount: number
  width: number
}

const initialOptionsState: OptionsState = {
  activeTab: 0,
  cfgScale: 7.5,
  codeformerFidelity: 0.75,
  currentTheme: 'dark',
  facetoolStrength: 0.8,
  facetoolType: 'gfpgan',
  height: 512,
  hiresFix: false,
  img2imgStrength: 0.75,
  infillMethod: 'patchmatch',
  isLightBoxOpen: false,
  iterations: 1,
  maskPath: '',
  optionsPanelScrollPosition: 0,
  perlin: 0,
  prompt: '',
  sampler: 'k_lms',
  seamBlur: 16,
  seamless: false,
  seamSize: 96,
  seamSteps: 10,
  seamStrength: 0.7,
  seed: 0,
  seedWeights: '',
  shouldFitToWidthHeight: true,
  shouldGenerateVariations: false,
  shouldHoldOptionsPanelOpen: false,
  shouldLoopback: false,
  shouldPinOptionsPanel: true,
  shouldRandomizeSeed: true,
  shouldRunESRGAN: false,
  shouldRunFacetool: false,
  shouldShowOptionsPanel: true,
  showDualDisplay: true,
  steps: 50,
  threshold: 0,
  tileSize: 32,
  upscalingLevel: 4,
  upscalingStrength: 0.75,
  variationAmount: 0.1,
  width: 512,
}

const initialState: OptionsState = initialOptionsState

// export const optionsSlice = createSlice({
// 	name: 'options',
// 	initialState,
// 	reducers: {
// 		setMaskPath: (state, action: PayloadAction<string>) => {
// 			state.maskPath = action.payload;
// 		},
// 		setShowDualDisplay: (state, action: PayloadAction<boolean>) => {
// 			state.showDualDisplay = action.payload;
// 		},
// 		setInitialImage: (state, action: PayloadAction<Image | string>) => {
// 			state.initialImage = action.payload;
// 		},
// 		clearInitialImage: state => {
// 			state.initialImage = undefined;
// 		},
// 		setInfillMethod: (state, action: PayloadAction<string>) => {
// 			state.infillMethod = action.payload;
// 		},
// 	},
// });

// export const {
// 	clearInitialImage,
// 	setInfillMethod,
// 	setInitialImage,
// 	setMaskPath,
// 	setShowDualDisplay,
// } = optionsSlice.actions;

// export default optionsSlice.reducer;
