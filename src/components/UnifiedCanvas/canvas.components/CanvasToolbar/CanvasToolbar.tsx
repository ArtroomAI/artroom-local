import {
	ButtonGroup,
	createStandaloneToast,
	Flex,
	useColorModeValue,
	Box,
	HStack,
	Text,
} from '@chakra-ui/react';
import {
	FaArrowsAlt,
	FaCopy,
	FaCrosshairs,
	FaPlay,
	FaSave,
	FaTrash,
	FaUpload,
	FaCheck,
	FaPlus,
} from 'react-icons/fa';
import { RiDragDropLine } from 'react-icons/ri';
import { useHotkeys } from 'react-hotkeys-hook';
import { FC, ChangeEvent } from 'react';
import { useRecoilState, useSetRecoilState, useRecoilValue } from 'recoil';
import { useImageUploader } from '../../hooks';
import {
	toolSelector,
	layerAtom,
	setIsMaskEnabledAction,
	resetCanvasAction,
	resetCanvasViewAction,
	resizeAndScaleCanvasAction,
	// shouldCropToBoundingBoxOnSaveAtom,
	isStagingSelector,
	stageScaleAtom,
	boundingBoxCoordinatesAtom,
	boundingBoxDimensionsAtom,
	layerStateAtom,
	stageCoordinatesAtom,
} from '../../atoms/canvas.atoms';
import { imageSettingsState } from '../../../../atoms/atoms';
import { isProcessingAtom } from '../../atoms/system.atoms';
import { Select, IconButton } from '../../components';
import {
	CanvasLayer,
	isCanvasMaskLine,
	LAYER_NAMES_DICT,
} from '../../atoms/canvasTypes';
import { CanvasToolChooserOptions } from './CanvasToolChooserOptions';
import {
	// copyImage,
	generateMask,
	getCanvasBaseLayer,
	layerToDataURL,
} from '../../util';
import { CanvasMaskOptions } from './CanvasMaskOptions';
import { CanvasSettingsButtonPopover } from './CanvasSettingsButtonPopover';
import { CanvasRedoButton } from './CanvasRedoButton';
import { CanvasUndoButton } from './CanvasUndoButton';
import axios from 'axios';
import { TbResize } from 'react-icons/tb';
import { useEventEmitter } from '../../../../helpers';
// import path from 'path';

const ARTROOM_URL = import.meta.env.VITE_ARTROOM_URL;

export const CanvasOutpaintingControls: FC = () => {
	const canvasBaseLayer = getCanvasBaseLayer();

	const [tool, setTool] = useRecoilState(toolSelector);
	const [layer, setLayer] = useRecoilState(layerAtom);
	const [isMaskEnabled, setIsMaskEnabled] = useRecoilState(
		setIsMaskEnabledAction,
	);
	const resetCanvas = useSetRecoilState(resetCanvasAction);
	const resetCanvasView = useSetRecoilState(resetCanvasViewAction);
	const resizeAndScaleCanvas = useSetRecoilState(resizeAndScaleCanvasAction);
	// const shouldCropToBoundingBoxOnSave = useRecoilValue(
	// 	shouldCropToBoundingBoxOnSaveAtom,
	// );
	const isProcessing = useRecoilValue(isProcessingAtom);
	const isStaging = useRecoilValue(isStagingSelector);

	const { openUploader } = useImageUploader();

	//For Generation
	const stageScale = useRecoilValue(stageScaleAtom);
	const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);
	const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);
	const layerState = useRecoilValue(layerStateAtom);
	const stageCoordinates = useRecoilValue(stageCoordinatesAtom);
	const [imageSettings, setImageSettings] =
		useRecoilState(imageSettingsState);
	const { ToastContainer, toast } = createStandaloneToast();

	const { emit } = useEventEmitter();

	useHotkeys(
		['v'],
		() => {
			handleSelectMoveTool();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[],
	);

	useHotkeys(
		['k'],
		() => {
			handleSelectMoveBoundingBoxTool();
		},
		{ enabled: () => true, preventDefault: true },
		[],
	);

	useHotkeys(
		['g'],
		() => {
			handleSelectTransformTool();
		},
		{ enabled: () => true, preventDefault: true },
		[],
	);

	useHotkeys(
		['r'],
		() => {
			handleResetCanvasView();
		},
		{
			enabled: () => true,
			preventDefault: true,
		},
		[canvasBaseLayer],
	);

	useHotkeys(
		['shift+s'],
		() => {
			handleSaveToGallery();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[canvasBaseLayer, isProcessing],
	);

	useHotkeys(
		['meta+c', 'ctrl+c'],
		() => {
			handleCopyImageToClipboard();
		},
		{
			enabled: () => !isStaging,
			preventDefault: true,
		},
		[canvasBaseLayer, isProcessing],
	);

	const handleSelectMoveTool = () => setTool('move');

	const handleSelectMoveBoundingBoxTool = () => setTool('moveBoundingBox');

	const handleSelectTransformTool = () => setTool('transform');

	const handleResetCanvasView = () => {
		const canvasBaseLayer = getCanvasBaseLayer();
		if (!canvasBaseLayer) return;
		const clientRect = canvasBaseLayer.getClientRect({
			skipTransform: true,
		});
		console.log(clientRect, 'clientRect');
		resetCanvasView({
			contentRect: clientRect,
		});
	};

	const handleResetCanvas = () => {
		resetCanvas();
		resizeAndScaleCanvas();
	};

	const handleSaveToGallery = () => {
		const canvasBaseLayer = getCanvasBaseLayer();

		if (canvasBaseLayer) {
			const { dataURL, boundingBox: originalBoundingBox } =
				layerToDataURL(canvasBaseLayer, stageScale, stageCoordinates);
		}

		const timestamp = new Date().getTime();
		// const imagePath = path.join(
		// 	imageSettings.image_save_path,
		// 	imageSettings.batch_name,
		// 	timestamp + '.jpg',
		// );
		// console.log(imagePath);
		// window.api.saveFromDataURL(JSON.stringify({ dataURL, imagePath }));
		toast({
			title: 'Image Saved',
			status: 'success',
			duration: 1500,
			isClosable: true,
		});
	};

	const handleCopyImageToClipboard = () => {
		const canvasBaseLayer = getCanvasBaseLayer();

		if (canvasBaseLayer) {
			const { dataURL, boundingBox: originalBoundingBox } =
				layerToDataURL(canvasBaseLayer, stageScale, stageCoordinates);
		}

		// window.api.copyToClipboard(dataURL);
		toast({
			title: 'Image Copied',
			status: 'success',
			duration: 1500,
			isClosable: true,
		});
	};

	const handleRunInpainting = async () => {
		const canvasBaseLayer = getCanvasBaseLayer();

		const boundingBox = {
			...boundingBoxCoordinates,
			...boundingBoxDimensions,
		};
		const maskDataURL = generateMask(
			isMaskEnabled ? layerState.objects.filter(isCanvasMaskLine) : [],
			boundingBox,
		);

		const tempScale = canvasBaseLayer?.scale();

		canvasBaseLayer?.scale({
			x: 1 / stageScale,
			y: 1 / stageScale,
		});

		const absPos = canvasBaseLayer?.getAbsolutePosition();

		const imageDataURL = canvasBaseLayer?.toDataURL({
			x: boundingBox.x + (absPos?.x || 0),
			y: boundingBox.y + (absPos?.y || 0),
			width: boundingBox.width,
			height: boundingBox.height,
		});

		canvasBaseLayer?.scale(tempScale);
		const body = {
			...imageSettings,
			init_image: imageDataURL,
			mask_image: maskDataURL,
		};

		// axios
		// 	.post(`${ARTROOM_URL}/add_to_queue`, body, {
		// 		headers: { 'Content-Type': 'application/json' },
		// 	})
		// 	.then(result => {
		// 		console.log(result);
		// 	});
	};

	const handleChangeLayer = (e: ChangeEvent<HTMLSelectElement>) => {
		const newLayer = e.target.value as CanvasLayer;
		setLayer(newLayer);
		if (newLayer === 'mask' && !isMaskEnabled) {
			setIsMaskEnabled(true);
		}
		if (
			(newLayer === 'mask' || newLayer === 'base') &&
			tool === 'transform'
		) {
			setTool('brush');
		}
	};

	return (
		<Flex alignItems="center" justifyContent="space-between" width="100%">
			<Box flex={0.2} />
			<Flex alignItems="center" columnGap="0.5rem" flex={0.6}>
				<Select
					tooltip="Layer (Q)"
					tooltipProps={{ hasArrow: true, placement: 'top' }}
					value={layer}
					validValues={[
						...[...layerState.images].reverse().map(elem => ({
							key: elem.name,
							value: elem.id,
						})),
						...LAYER_NAMES_DICT,
					]}
					onChange={handleChangeLayer}
					isDisabled={isStaging}
				/>

				<CanvasMaskOptions />
				<CanvasSettingsButtonPopover />

				<CanvasToolChooserOptions />

				<ButtonGroup isAttached>
					<IconButton
						aria-label="Move Bounding Box (K)"
						onClick={handleSelectMoveBoundingBoxTool}
						icon={<RiDragDropLine size={20} />}
						tooltip="Move Bounding Box"
						aria-selected={tool === 'moveBoundingBox' || isStaging}
					/>
					<IconButton
						aria-label="Reset View (R)"
						tooltip="Reset View (R)"
						icon={<FaCrosshairs />}
						onClick={handleResetCanvasView}
					/>
					{layer !== 'base' && layer !== 'mask' ? (
						<>
							<IconButton
								aria-label="Move Tool (V)"
								tooltip="Move Tool (V)"
								icon={<FaArrowsAlt />}
								aria-selected={tool === 'move' || isStaging}
								onClick={handleSelectMoveTool}
							/>
							<IconButton
								icon={<TbResize />}
								aria-label="Transform (G)"
								tooltip="Transform (G)"
								onClick={handleSelectTransformTool}
								aria-selected={
									tool === 'transform' || isStaging
								}
							/>
						</>
					) : null}
				</ButtonGroup>
				<ButtonGroup isAttached>
					<IconButton
						aria-label="Save to Gallery (Shift+S)"
						tooltip="Save to Gallery (Shift+S)"
						icon={<FaSave />}
						onClick={handleSaveToGallery}
						isDisabled={isStaging}
					/>
					<IconButton
						aria-label="Copy to Clipboard (Cmd/Ctrl+C)"
						tooltip="Copy to Clipboard (Cmd/Ctrl+C)"
						icon={<FaCopy />}
						onClick={handleCopyImageToClipboard}
						isDisabled={isStaging}
					/>
					<IconButton
						aria-label="Upload"
						tooltip="Upload"
						icon={<FaUpload />}
						onClick={openUploader}
						isDisabled={isStaging}
					/>
					<IconButton
						aria-label="Clear Canvas"
						tooltip="Clear Canvas"
						icon={<FaTrash />}
						onClick={handleResetCanvas}
						bg={useColorModeValue('#d53131', '#b62e2e')}
						sx={{ _hover: { bg: 'red.700 !important' } }}
						isDisabled={isStaging}
					/>
				</ButtonGroup>

				<ButtonGroup isAttached>
					<CanvasUndoButton />
					<CanvasRedoButton />
				</ButtonGroup>

				<ButtonGroup isAttached>
					<IconButton
						aria-label="Run (Alt+R)"
						tooltip="Run (Alt+R)"
						icon={<FaPlay />}
						onClick={handleRunInpainting}
						isDisabled={isStaging}
					/>
				</ButtonGroup>
			</Flex>
			<HStack flex={0.2} justifyContent="flex-end">
				{tool === 'transform' ? (
					<>
						<Text>Transform:</Text>
						<ButtonGroup isAttached>
							<IconButton
								aria-label="Accept"
								tooltip="Accept"
								icon={<FaCheck />}
								onClick={() => emit('acceptTransform')}
								isDisabled={isStaging}
							/>
							<IconButton
								aria-label="Accept"
								tooltip="Accept"
								icon={
									<FaPlus
										style={{ transform: 'rotate(45deg)' }}
									/>
								}
								onClick={() => emit('rejectTransform')}
								isDisabled={isStaging}
							/>
						</ButtonGroup>
					</>
				) : null}
			</HStack>
		</Flex>
	);
};
