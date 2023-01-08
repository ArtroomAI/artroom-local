/**
 * Types for images, the things they are made of, and the things
 * they make up.
 *
 * Generated images are txt2img and img2img images. They may have
 * had additional postprocessing done on them when they were first
 * generated.
 *
 * Postprocessed images are images which were not generated here
 * but only postprocessed by the app. They only get postprocessing
 * metadata and have a different image type, e.g. 'esrgan' or
 * 'gfpgan'.
 */

import { IRect } from "konva/lib/types";
import { ExpandedIndex, UseToastOptions } from "@chakra-ui/react";
import {
	InProgressImageType,
	LogEntry,
} from "./examples/system.store.example/systemSlice.example";
import { FACETOOL_TYPES } from "./constants";

/**
 * TODO:
 * Once an image has been generated, if it is postprocessed again,
 * additional postprocessing steps are added to its postprocessing
 * array.
 *
 * TODO: Better documentation of types.
 */

export declare type PromptItem = {
	prompt: string;
	weight: number;
};

export declare type Prompt = Array<PromptItem>;

export declare type SeedWeightPair = {
	seed: number;
	weight: number;
};

export declare type SeedWeights = Array<SeedWeightPair>;

// Superset of  generated image metadata types.
export declare type GeneratedImageMetadata = Txt2ImgMetadata | Img2ImgMetadata;

// All post processed images contain these metadata.
export declare type CommonPostProcessedImageMetadata = {
	orig_path: string;
	orig_hash: string;
};

// esrgan and gfpgan images have some unique attributes.
export declare type ESRGANMetadata = CommonPostProcessedImageMetadata & {
	type: "esrgan";
	scale: 2 | 4;
	strength: number;
};

export declare type FacetoolMetadata = CommonPostProcessedImageMetadata & {
	type: "gfpgan" | "codeformer";
	strength: number;
	fidelity?: number;
};

// Superset of all postprocessed image metadata types..
export declare type PostProcessedImageMetadata =
	| ESRGANMetadata
	| FacetoolMetadata;

// Metadata includes the system config and image metadata.
export declare type Metadata = SystemGenerationMetadata & {
	image: GeneratedImageMetadata | PostProcessedImageMetadata;
};

// An Image has a UUID, url, modified timestamp, width, height and maybe metadata
export declare type Image = {
	uuid: string;
	url: string;
	thumbnail: string;
	mtime: number;
	metadata?: Metadata;
	width: number;
	height: number;
	category: GalleryCategory;
	isBase64?: boolean;
	dreamPrompt?: "string";
};

/**
 * Types related to the system status.
 */

// This represents the processing status of the backend.
export declare type SystemStatus = {
	isProcessing: boolean;
	currentStep: number;
	totalSteps: number;
	currentIteration: number;
	totalIterations: number;
	currentStatus: string;
	currentStatusHasSteps: boolean;
	hasError: boolean;
};

export declare type SystemGenerationMetadata = {
	model: string;
	model_weights?: string;
	model_id?: string;
	model_hash: string;
	app_id: string;
	app_version: string;
};

export declare type SystemConfig = SystemGenerationMetadata & {
	model_list: ModelList;
	infill_methods: string[];
};

export declare type ModelStatus = "active" | "cached" | "not loaded";

export declare type Model = {
	status: ModelStatus;
	description: string;
};

export declare type ModelList = Record<string, Model>;

/**
 * These types type data received from the server via socketio.
 */

export declare type ModelChangeResponse = {
	model_name: string;
	model_list: ModelList;
};

export declare type ImageResultResponse = Omit<Image, "uuid"> & {
	boundingBox?: IRect;
	generationMode: "unifiedCanvas";
};

export declare type ImageUploadResponse = {
	// image: Omit<Image, 'uuid' | 'metadata' | 'category'>;
	url: string;
	mtime: number;
	width: number;
	height: number;
	thumbnail: string;
	// bbox: [number, number, number, number];
};

export declare type ErrorResponse = {
	message: string;
	additionalData?: string;
};

export declare type GalleryImagesResponse = {
	images: Array<Omit<Image, "uuid">>;
	areMoreImagesAvailable: boolean;
	category: GalleryCategory;
};

export declare type ImageDeletedResponse = {
	uuid: string;
	url: string;
	category: GalleryCategory;
};

export interface SystemState extends SystemStatus, SystemConfig {
	shouldDisplayInProgressType: InProgressImageType;
	log: Array<LogEntry>;
	isGFPGANAvailable: boolean;
	isESRGANAvailable: boolean;
	isConnected: boolean;
	socketId: string;
	openAccordions: ExpandedIndex;
	currentStep: number;
	totalSteps: number;
	currentIteration: number;
	totalIterations: number;
	currentStatus: string;
	currentStatusHasSteps: boolean;
	shouldDisplayGuides: boolean;
	wasErrorSeen: boolean;
	isCancelable: boolean;
	saveIntermediatesInterval: number;
	enableImageDebugging: boolean;
	toastQueue: UseToastOptions[];
}

export interface OptionsState {
	activeTab: number;
	cfgScale: number;
	codeformerFidelity: number;
	currentTheme: string;
	facetoolStrength: number;
	facetoolType: FacetoolType;
	height: number;
	hiresFix: boolean;
	img2imgStrength: number;
	infillMethod: string;
	initialImage?: Image | string; // can be an Image or url
	isLightBoxOpen: boolean;
	iterations: number;
	maskPath: string;
	optionsPanelScrollPosition: number;
	perlin: number;
	prompt: string;
	sampler: string;
	seamBlur: number;
	seamless: boolean;
	seamSize: number;
	seamSteps: number;
	seamStrength: number;
	seed: number;
	seedWeights: string;
	shouldFitToWidthHeight: boolean;
	shouldGenerateVariations: boolean;
	shouldHoldOptionsPanelOpen: boolean;
	shouldLoopback: boolean;
	shouldPinOptionsPanel: boolean;
	shouldRandomizeSeed: boolean;
	shouldRunESRGAN: boolean;
	shouldRunFacetool: boolean;
	shouldShowOptionsPanel: boolean;
	showDualDisplay: boolean;
	steps: number;
	threshold: number;
	tileSize: number;
	upscalingLevel: UpscalingLevel;
	upscalingStrength: number;
	variationAmount: number;
	width: number;
}

export type UpscalingLevel = 2 | 4;

export type FacetoolType = typeof FACETOOL_TYPES[number];
