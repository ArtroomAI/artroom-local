import { FC, ChangeEvent } from 'react';
import { Flex, Checkbox } from '@chakra-ui/react';
import { FaWrench } from 'react-icons/fa';
import { useHotkeys } from 'react-hotkeys-hook';
import { useRecoilState } from 'recoil';
import { Popover, IconButton } from '../../components';
import { EmptyTempFolderButtonModal } from '../ClearTempFolderButtonModal';
import { ClearCanvasHistoryButtonModal } from '../ClearCanvasHistoryButtonModal';
import {
	shouldDarkenOutsideBoundingBoxAtom,
	shouldSnapToGridAtom,
	shouldAutoSaveAtom,
	shouldShowGridAtom,
	shouldShowCanvasDebugInfoAtom,
	shouldRestrictStrokesToBoxAtom,
	shouldCropToBoundingBoxOnSaveAtom,
	shouldShowIntermediatesAtom,
} from '../../atoms/canvas.atoms';

export const CanvasSettingsButtonPopover: FC = () => {
	const [shouldDarkenOutsideBoundingBox, setShouldDarkenOutsideBoundingBox] =
		useRecoilState(shouldDarkenOutsideBoundingBoxAtom);
	const [shouldSnapToGrid, setShouldSnapToGrid] =
		useRecoilState(shouldSnapToGridAtom);
	const [shouldAutoSave, setShouldAutoSave] =
		useRecoilState(shouldAutoSaveAtom);
	const [shouldShowGrid, setShouldShowGrid] =
		useRecoilState(shouldShowGridAtom);
	const [shouldShowCanvasDebugInfo, setShouldShowCanvasDebugInfo] =
		useRecoilState(shouldShowCanvasDebugInfoAtom);
	const [shouldRestrictStrokesToBox, setShouldRestrictStrokesToBox] =
		useRecoilState(shouldRestrictStrokesToBoxAtom);
	const [shouldCropToBoundingBoxOnSave, setShouldCropToBoundingBoxOnSave] =
		useRecoilState(shouldCropToBoundingBoxOnSaveAtom);
	const [shouldShowIntermediates, setShouldShowIntermediates] =
		useRecoilState(shouldShowIntermediatesAtom);

	useHotkeys(
		['n'],
		() => {
			setShouldSnapToGrid(!shouldSnapToGrid);
		},
		{
			enabled: true,
			preventDefault: true,
		},
		[shouldSnapToGrid],
	);

	const handleChangeShouldSnapToGrid = (e: ChangeEvent<HTMLInputElement>) =>
		setShouldSnapToGrid(e.target.checked);

	return (
		<Popover
			trigger="hover"
			triggerComponent={
				<IconButton
					tooltip="Canvas Settings"
					aria-label="Canvas Settings"
					icon={<FaWrench />}
				/>
			}>
			<Flex direction="column" gap="0.5rem">
				<Checkbox
					isChecked={shouldShowIntermediates}
					onChange={e =>
						setShouldShowIntermediates(e.target.checked)
					}>
					Show Intermediates
				</Checkbox>
				<Checkbox
					isChecked={shouldShowGrid}
					onChange={e => setShouldShowGrid(e.target.checked)}>
					Show Grid
				</Checkbox>
				<Checkbox
					isChecked={shouldSnapToGrid}
					onChange={handleChangeShouldSnapToGrid}>
					Snap to Grid
				</Checkbox>
				<Checkbox
					isChecked={shouldDarkenOutsideBoundingBox}
					onChange={e =>
						setShouldDarkenOutsideBoundingBox(e.target.checked)
					}>
					Darken Outside Selection
				</Checkbox>
				<Checkbox
					isChecked={shouldAutoSave}
					onChange={e => setShouldAutoSave(e.target.checked)}>
					Auto Save to Gallery
				</Checkbox>
				<Checkbox
					isChecked={shouldCropToBoundingBoxOnSave}
					onChange={e =>
						setShouldCropToBoundingBoxOnSave(e.target.checked)
					}>
					Save Box Region Only
				</Checkbox>
				{/* <Checkbox
					isChecked={shouldRestrictStrokesToBox}
					onChange={e =>
						setShouldRestrictStrokesToBox(e.target.checked)
					}>
					Limit Strokes to Box
				</Checkbox> */}
				<Checkbox
					isChecked={shouldShowCanvasDebugInfo}
					onChange={e =>
						setShouldShowCanvasDebugInfo(e.target.checked)
					}>
					Show Canvas Debug Info
				</Checkbox>
				<ClearCanvasHistoryButtonModal />
				<EmptyTempFolderButtonModal />
			</Flex>
		</Popover>
	);
};
